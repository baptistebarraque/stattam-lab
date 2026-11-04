"""Module autonome du notebook `11_ventes_final.ipynb` (synthèse finale, Kaggle Store Sales / Favorita).

Ce fichier et le notebook suffisent : ils ne dépendent que de TAM, numpy, pandas, scipy et matplotlib.
Les données Kaggle (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`)
sont cherchées dans le dossier donné par `set_data_dir`, sinon dans la variable d'environnement `STORE_SALES_DIR`,
sinon dans `data_store_sales/store-sales-time-series-forecasting` (dossier courant ou un de ses parents).

Contenu : données (chargement, tableau origine × horizon), évaluation (RMSLE, écart de MSLE par
date), classes de demande (`idclass`, équivalent de `tsintermittent::idclass`), et les figures du notebook.
Les appels à TAM (ajustement, lissage, explicabilité, diagnostics, résidus) restent dans le notebook.
"""
from __future__ import annotations

import contextlib
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

with contextlib.redirect_stdout(io.StringIO()):          # bannière matérielle de TAM
    import tam

__all__ = ['set_data_dir', 'load_raw', 'Favorita', 'quiet', 'rmsle', 'daily_msle', 'idclass',
           'F1', 'F2', 'F_S', 'F3', 'plot_exploration', 'build_frames', 'evaluate', 'versus', 'benchmark_table',
           'plot_benchmark', 'family_rows', 'plot_residuals']

DAYS = ['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim']
HORIZON = 16

#: Protocole : origines hebdomadaires le mardi (comme le test) ; `train*` : origines dont les cibles précèdent la validation
SS_PERIODS = {
    'trainA': ('2015-08-04', '2017-02-12'),   # cibles jusqu'au 28/02/2017
    'valA': ('2017-03-07', '2017-04-25'),     # 8 origines, cibles du 08/03 au 11/05/2017
    'trainB': ('2015-08-04', '2017-05-21'),   # cibles jusqu'au 06/06/2017
    'valB': ('2017-06-06', '2017-07-25'),     # 8 origines, cibles du 07/06 au 10/08/2017
}


# ===============================================================================================================
# 1. Données
# ===============================================================================================================
_DATA_DIR = None
_SUBDIR = Path('data_store_sales') / 'store-sales-time-series-forecasting'


def set_data_dir(path) -> None:
    """Le dossier qui contient les fichiers CSV de la compétition."""
    global _DATA_DIR
    _DATA_DIR = Path(path)


def _data_dir() -> Path:
    if _DATA_DIR is not None:
        return _DATA_DIR
    if os.environ.get('STORE_SALES_DIR'):
        return Path(os.environ['STORE_SALES_DIR'])
    here = Path(__file__).resolve().parent
    for base in (Path.cwd(), *Path.cwd().parents, here, *here.parents):
        if (base / _SUBDIR / 'train.csv').exists():
            return base / _SUBDIR
        if (base / 'train.csv').exists() and (base / 'holidays_events.csv').exists():
            return base
    raise FileNotFoundError("Données introuvables : appeler set_data_dir('<dossier des CSV Kaggle>')")


def load_raw():
    """Les fichiers bruts : (train, test, stores, oil, holidays, transactions), dates converties."""
    d = _data_dir()

    def read(name, **kw):
        return pd.read_csv(d / name, **kw)

    return (read('train.csv', parse_dates=['date']), read('test.csv', parse_dates=['date']), read('stores.csv'),
            read('oil.csv', parse_dates=['date']), read('holidays_events.csv', parse_dates=['date']),
            read('transactions.csv', parse_dates=['date']))


def origins_between(start: str, end: str) -> list:
    """Les mardis de `start` à `end` inclus (les origines du protocole)."""
    days = pd.date_range(start, end, freq='D')
    return list(days[days.dayofweek == 1])


class Favorita:
    """Les ventes en tableau large (jours × séries) et la fabrique des lignes origine × horizon.

    Attributs : `series` (les 1 782 séries magasin × famille, avec `fam_id`), `days`, `sales` (ventes brutes), `Y`
    (log1p des ventes), `P` (promotions, test compris), `cal` (calendrier par jour), `local` (jour férié local ou
    régional, jours × magasins), `oil`.
    """

    def __init__(self):
        train, test, stores, oil, holidays, _ = load_raw()
        both = pd.concat([train, test], ignore_index=True)
        self.series = (both[['store_nbr', 'family']].drop_duplicates().sort_values(['store_nbr', 'family'])
                       .reset_index(drop=True).merge(stores, on='store_nbr', how='left'))
        families = sorted(self.series['family'].unique())
        self.series['fam_id'] = self.series['family'].map({f: i for i, f in enumerate(families)})
        self.series['store_id'] = self.series['store_nbr'] - 1
        self.series['type_id'] = self.series['type'].map({t: i for i, t in enumerate('ABCDE')})
        self.series['city_id'] = pd.Categorical(self.series['city']).codes
        self.days = pd.date_range(train['date'].min(), test['date'].max(), freq='D')
        key = pd.MultiIndex.from_frame(self.series[['store_nbr', 'family']])

        def wide(col, frame):
            return (frame.pivot_table(index='date', columns=['store_nbr', 'family'], values=col, aggfunc='sum')
                    .reindex(index=self.days, columns=key))

        self.sales = wide('sales', train).to_numpy(float)
        self.Y = np.log1p(self.sales)
        self.P = wide('onpromotion', both).to_numpy(float)
        self.last_day = train['date'].max()
        self.day_index = {d: i for i, d in enumerate(self.days)}
        self.oil = oil.set_index('date')['dcoilwtico'].reindex(self.days).ffill().bfill().to_numpy(float)
        self._build_calendar(holidays, stores)

    def _build_calendar(self, holidays: pd.DataFrame, stores: pd.DataFrame) -> None:
        h = holidays.copy()
        off = ((h['type'].isin(['Holiday', 'Additional', 'Bridge', 'Transfer'])) & ~h['transferred'])
        nat = h[off & (h['locale'] == 'National')]
        cal = pd.DataFrame(index=self.days)
        cal['dow'] = cal.index.dayofweek
        cal['dom'] = cal.index.day
        cal['payday'] = ((cal['dom'] == 15) | cal.index.is_month_end).astype(int)     # paie le 15 et en fin de mois
        cal['nat_hol'] = cal.index.isin(nat.loc[nat['type'] != 'Additional', 'date']).astype(int)
        cal['nat_add'] = cal.index.isin(nat.loc[nat['type'] == 'Additional', 'date']).astype(int)
        cal['newyear'] = ((cal.index.month == 1) & (cal.index.day == 1)).astype(int)
        self.cal = cal
        loc = h[off & (h['locale'] != 'National')]                    # fériés de la ville ou de la province
        local = np.zeros((len(self.days), len(stores)), dtype=int)
        for _, r in loc.iterrows():
            if r['date'] not in self.day_index:
                continue
            col = stores['city'] if r['locale'] == 'Local' else stores['state']
            local[self.day_index[r['date']], np.flatnonzero((col == r['locale_name']).to_numpy())] = 1
        self.local = local

    def frame(self, origins, horizons=range(1, HORIZON + 1), active_only: bool = True) -> pd.DataFrame:
        """Le tableau origine × horizon × série pour une liste d'origines.

        Historique (jusqu'à l'origine incluse) : `last`, `lvl7`, `lvl14`, `lvl28`, `lvl56` (moyennes de y = log1p(ventes)),
        `dow8` (moyenne de y sur les 8 derniers mêmes jours de semaine que la cible), `zero28` (part de jours sans vente
        sur 28 jours), `promo28`. Connu pour la cible : `promo`, `promo_rel` = `promo - promo28`, calendrier, `loc_hol`, `h`.
        `active` : la série a vendu dans les 56 jours avant l'origine. Les lignes d'un magasin pas encore ouvert et les
        cibles du 25 décembre (magasins fermés) sont retirées.
        """
        blocks = []
        S = len(self.series)
        Y0 = self.Y
        logP = np.log1p(np.nan_to_num(self.P, nan=0.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # moyenne d'une série sans historique : NaN
            for o in origins:
                io_ = self.day_index[pd.Timestamp(o)]
                lv = {k: np.nanmean(Y0[io_ - k + 1: io_ + 1], axis=0) for k in (1, 7, 14, 28, 56)}
                zero28 = np.nanmean(Y0[io_ - 27: io_ + 1] == 0, axis=0)
                active = np.nansum(Y0[io_ - 55: io_ + 1], axis=0) > 0
                store_open = pd.Series(active).groupby(self.series['store_nbr'].to_numpy()).transform('any').to_numpy()
                promo28 = logP[io_ - 27: io_ + 1].mean(axis=0)
                for h in horizons:
                    idx = io_ + h
                    same = [idx - 7 * j for j in range(1, 20) if idx - 7 * j <= io_][:8]
                    blocks.append(pd.DataFrame({
                        'origin': np.repeat(self.days[io_], S), 'date': np.repeat(self.days[idx], S), 'h': h,
                        'series': np.arange(S),
                        'last': lv[1], 'lvl7': lv[7], 'lvl14': lv[14], 'lvl28': lv[28], 'lvl56': lv[56],
                        'dow8': np.nanmean(Y0[same], axis=0), 'zero28': zero28, 'active': active, 'store_open': store_open,
                        'promo': logP[idx], 'promo28': promo28, 'oil_o': self.oil[io_],
                        'loc_hol': self.local[idx, self.series['store_id'].to_numpy()],
                        'y': Y0[idx] if idx < len(Y0) else np.nan,
                    }))
        df = pd.concat(blocks, ignore_index=True)
        keep = (df['y'].notna() | (df['date'] > self.last_day)) & df['store_open']
        if active_only:
            keep &= df['active']
        df = df[keep].reset_index(drop=True)
        df['promo_rel'] = df['promo'] - df['promo28']
        cal = self.cal.reindex(df['date']).reset_index(drop=True)
        for c in cal.columns:
            df[c] = cal[c].to_numpy()
        ser = self.series.iloc[df['series'].to_numpy()].reset_index(drop=True)
        for c in ('store_nbr', 'family', 'fam_id', 'store_id', 'type_id', 'cluster', 'city_id'):
            df[c] = ser[c].to_numpy()
        return df

    def period(self, name: str, **kwargs) -> pd.DataFrame:
        """Le tableau d'une période du protocole (`SS_PERIODS`)."""
        return self.frame(origins_between(*SS_PERIODS[name]), **kwargs)


# ===============================================================================================================
# 2. Évaluation
# ===============================================================================================================
@contextlib.contextmanager
def quiet():
    """Masque les impressions de TAM (bannière matérielle, journal de l'auto-fit)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def rmsle(y_log, p_log) -> float:
    """RMSLE quand cible et prévision sont déjà en log1p (prévision ramenée à >= 0)."""
    y, p = np.asarray(y_log, float), np.clip(np.asarray(p_log, float), 0.0, None)
    ok = ~np.isnan(y)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def daily_msle(frame: pd.DataFrame, pred, col: str = 'y') -> pd.Series:
    """Erreur quadratique moyenne en log par date cible (toutes origines et séries confondues)."""
    e = (frame[col].to_numpy(float) - np.clip(np.asarray(pred, float), 0.0, None)) ** 2
    return pd.Series(e, index=frame['date'].to_numpy()).groupby(level=0).mean()


def _predict(model, data: pd.DataFrame) -> np.ndarray:
    """Prévision d'un `StaticTAM` dans l'ordre des lignes de `data`."""
    with quiet():
        out = model.predict(data)
    return out[f'Estimated{model.target_col_}'].to_numpy(float)


# ===============================================================================================================
# 3. Classes de demande (fonction de R absente de TAM)
# ===============================================================================================================
def idclass(Y: np.ndarray, adi_cut: float = 1.32, cv2_cut: float = 0.49) -> pd.DataFrame:
    """Classe de demande de chaque série : régulière, erratique, intermittente, grumeleuse ou jamais vendue.

    Imite `tsintermittent::idclass(type = "SBC")` : ADI (intervalle moyen entre deux ventes non nulles) et CV² (carré
    du coefficient de variation des ventes non nulles), seuils 1,32 et 0,49.
    Référence : Syntetos, Boylan et Croston (2005), *Journal of the Operational Research Society*, 56(5), 495-503.
    `Y` : ventes brutes (jours × séries), NaN ignorés.
    """
    Y = np.asarray(Y, float)
    nonzero = (Y > 0) & ~np.isnan(Y)
    n_obs = (~np.isnan(Y)).sum(0)
    n_pos = nonzero.sum(0)
    adi = np.where(n_pos > 0, n_obs / np.maximum(n_pos, 1), np.inf)
    pos = np.where(nonzero, Y, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cv2 = (np.nanstd(pos, 0) / np.nanmean(pos, 0)) ** 2
    cls = np.select([n_pos == 0, (adi < adi_cut) & (cv2 < cv2_cut), adi < adi_cut, cv2 < cv2_cut],
                    ['jamais vendue', 'régulière', 'erratique', 'intermittente'], 'grumeleuse')
    return pd.DataFrame({'ADI': adi, 'CV2': cv2, 'classe': cls})


# ===============================================================================================================
# 4. Étapes et figures du notebook
# ===============================================================================================================
#: M1 : niveaux récents, profil de semaine, promotions, jours fériés, paie, horizon (tout linéaire sauf les promotions)
F1 = ("y ~ l(lvl7) + l(lvl14) + l(lvl28) + l(lvl56) + l(dow8) + c(dow, n_cat=7) + s(promo_rel) + l(zero28)"
      " + l(nat_hol) + l(nat_add) + l(loc_hol) + l(payday) + c(h, n_cat=17)")
#: M2 : M1 + jour de l'an, 1er et 2 du mois
F2 = F1 + " + l(newyear) + l(dom1) + l(dom2)"
#: M2 avec splines sur les niveaux (sans réseau) : sert au choix du lissage (GCV / REML)
F_S = F2.replace("l(lvl7)", "s(lvl7)").replace("l(dow8)", "s(dow8)")
#: M3 (modèle retenu) : + interaction niveau × horizon et réseau n()
F3 = F_S + " + te(l(lvl7), c(h, n_cat=17)) + n(lvl7, others=dow8|promo_rel|h, n_neurons=200)"


def plot_exploration(train_csv: pd.DataFrame) -> None:
    """Six vues des ventes brutes : tendance, cible, jour de semaine, jour du mois, intermittence, autocorrélation."""
    daily = train_csv.groupby('date')['sales'].sum()
    ratio = (daily / daily.rolling(28, center=True, min_periods=14).mean())['2016-06-01':]   # ventes du jour / niveau du mois
    recent = train_csv[train_csv['date'] >= '2016-06-01']
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes[0, 0].plot(daily.index, daily / 1e3, lw=0.5)
    axes[0, 0].axvline(pd.Timestamp('2016-04-16'), color='C3', ls='--', lw=0.8)
    axes[0, 0].set_title('ventes totales par jour (milliers) ; tirets : séisme')
    axes[0, 1].hist(np.log1p(recent['sales']), bins=80, color='0.5')
    axes[0, 1].set_title("y = log1p(ventes) d'une série un jour")
    axes[0, 2].bar(DAYS, ratio.groupby(ratio.index.dayofweek).mean())
    axes[0, 2].set_ylim(0.7, 1.35); axes[0, 2].axhline(1, color='k', lw=0.8)
    axes[0, 2].set_title('jour de semaine : ventes / niveau du mois')
    axes[1, 0].bar(range(1, 32), ratio.groupby(ratio.index.day).mean())
    axes[1, 0].set_ylim(0.8, 1.25); axes[1, 0].axhline(1, color='k', lw=0.8)
    axes[1, 0].set_title('jour du mois : ventes / niveau du mois')
    zeros = recent.groupby('family')['sales'].apply(lambda s: (s == 0).mean()).sort_values()
    axes[1, 1].barh(zeros.index, zeros.values); axes[1, 1].tick_params(axis='y', labelsize=5)
    axes[1, 1].set_title('part de jours sans vente, par famille')
    ylog = np.log1p(daily['2016-06-01':])
    axes[1, 2].bar(range(1, 29), [ylog.autocorr(k) for k in range(1, 29)])
    axes[1, 2].set_title('autocorrélation des ventes totales (log)'); axes[1, 2].set_xlabel('retard (jours)')
    plt.tight_layout(); plt.show()


def build_frames(F: Favorita, class_origin: str = '2017-06-06') -> dict:
    """Les tableaux origine × horizon (trainA, valA, trainB, valB).

    Une ligne = (série, origine = un mardi, horizon h = 1..16), historique figé à l'origine. On ajoute le 1er et le 2 du
    mois et la classe de demande (`idclass`), calculée sur l'année qui précède valB : elle sert à lire les résidus,
    pas au modèle.
    """
    io_ = F.day_index[pd.Timestamp(class_origin)]
    classes = idclass(F.sales[io_ - 364: io_ + 1])['classe'].to_numpy()

    def add_columns(frame):
        frame['dom1'] = (frame['dom'] == 1).astype(int)
        frame['dom2'] = (frame['dom'] == 2).astype(int)
        frame['classe'] = classes[frame['series'].to_numpy()]
        return frame

    return {name: add_columns(F.period(name, active_only=False)) for name in ('trainA', 'valA', 'trainB', 'valB')}


def evaluate(formula: str, d: dict, group_col: str = 'fam_id', **kw) -> dict:
    """Apprend sur trainA et trainB, prévoit valA et valB (log1p des ventes). Renvoie {'A', 'B', 'model' (appris sur trainB)}."""
    out = {}
    for ph in ('A', 'B'):
        model = tam.StaticTAM(formula=formula, group_col=group_col, **kw)
        with quiet():
            model.fit(d['train' + ph])
        out[ph] = _predict(model, d['val' + ph])
    out['model'] = model
    return out


def versus(pred: dict, ref: dict, d: dict) -> dict:
    """Écart moyen de MSLE par date cible (pred - ref) sur valA et valB : négatif = pred meilleur."""
    return {f'écart {ph}': float((daily_msle(d['val' + ph], pred[ph]) - daily_msle(d['val' + ph], ref[ph])).mean())
            for ph in ('A', 'B')}


def benchmark_table(preds: dict, d: dict, comparisons) -> pd.DataFrame:
    """RMSLE sur valA et valB, et l'écart de MSLE par date à une référence. `comparisons` : liste de (libellé, modèle, référence ou None)."""
    return pd.DataFrame({
        label: {'RMSLE valA': rmsle(d['valA']['y'], preds[name]['A']), 'RMSLE valB': rmsle(d['valB']['y'], preds[name]['B']),
                **(versus(preds[name], preds[ref], d) if ref else {})}
        for label, name, ref in comparisons}).T


def plot_benchmark(preds: dict, v: pd.DataFrame, ref: str = 'dow8', best: str = 'M3') -> None:
    """RMSLE sur valB par horizon, par date cible, et par famille (best contre ref)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name in preds:
        err = (v['y'] - preds[name]['B']) ** 2
        axes[0].plot(range(1, 17), np.sqrt(err.groupby(v['h']).mean()), 'o-', ms=3, label=name)
        e = np.sqrt(daily_msle(v, preds[name]['B']))
        axes[1].plot(e.index, e.values, lw=1, label=name)
    axes[0].set_title('RMSLE par horizon (valB)'); axes[0].set_xlabel('h'); axes[0].legend()
    axes[1].set_title('RMSLE par date cible (valB)'); axes[1].tick_params(axis='x', rotation=45)
    by_fam = pd.DataFrame({name: ((v['y'] - preds[name]['B']) ** 2).groupby(v['family']).mean() ** 0.5 for name in (ref, best)})
    axes[2].scatter(by_fam[ref], by_fam[best], s=10)
    axes[2].plot([0, by_fam.values.max()], [0, by_fam.values.max()], 'k', lw=0.8)
    for fam in by_fam[ref].nlargest(3).index:
        axes[2].annotate(fam, (by_fam.loc[fam, ref], by_fam.loc[fam, best]), fontsize=6)
    axes[2].set_xlabel(f'RMSLE {ref}'); axes[2].set_ylabel(f'RMSLE {best}')
    axes[2].set_title(f'par famille (sous la diagonale : {best} gagne)')
    plt.tight_layout(); plt.show()


def family_rows(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """Les lignes actives d'une famille, index remis à zéro (les fonctions graphiques de TAM superposent les groupes)."""
    return frame[(frame['family'] == family) & frame['active']].reset_index(drop=True)


def plot_residuals(v: pd.DataFrame) -> None:
    """Distribution des résidus contre une normale, QQ-plot, résidu moyen par date cible. `v` doit avoir une colonne 'r'."""
    r = v['r']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].hist(r, bins=200, density=True, color='0.5')
    grid = np.linspace(-3, 3, 200)
    axes[0].plot(grid, stats.norm.pdf(grid, r.mean(), r.std()), 'C3', label='normale de même écart-type')
    axes[0].set_xlim(-3, 3); axes[0].legend(fontsize=7); axes[0].set_title('distribution des résidus')
    sample = np.sort(r.sample(min(20000, len(r)), random_state=0))
    axes[1].plot(stats.norm.ppf((np.arange(len(sample)) + 0.5) / len(sample)) * sample.std(), sample, '.', ms=1)
    axes[1].plot([-3, 3], [-3, 3], 'k', lw=0.8); axes[1].set_title('QQ-plot contre la normale')
    by_date = v.groupby('date')['r'].mean()
    axes[2].bar(by_date.index, by_date.values, width=0.8); axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_title('résidu moyen par date cible (chocs communs)')
    plt.tight_layout(); plt.show()
