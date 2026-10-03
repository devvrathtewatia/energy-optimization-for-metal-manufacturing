"""Flatten every experiment record into tables.  python -m experiments.summarize"""
import pandas as pd

from src import config as C, experiment as E


def flat():
    rows = []
    for e in E.load_all():
        r = dict(dataset=e["dataset"], exp_id=e["exp_id"], model=e["model"], family=e["family"], complexity=e["complexity"], fit_seconds=e.get("fit_seconds"))
        for sp, subs in e["metrics"].items():
            for sub, ms in subs.items():
                for k, v in ms.items():
                    if k != "n":
                        r[f"{sp}.{sub}.{k}"] = v
        for sp, subs in e.get("metrics_std", {}).items():
            for sub, ms in subs.items():
                if sp == "val" and sub == "all":
                    r["val.all.mae_std_over_seeds"] = ms["mae"]
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = flat()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / "all_experiments_long.csv", index=False)
    for ds in ["ibarmia", "gmtk"]:
        s = df[df.dataset == ds].sort_values("val.all.mae")
        s[["exp_id", "model", "family", "val.all.mae", "val.all.rmse", "val.all.r2", "val.all.mape", "test.all.mae", "test.all.rmse", "test.all.r2", "test.all.mape"]].to_csv(C.TABLES / f"leaderboard_real_{ds}.csv", index=False)
    print(df.groupby("dataset").size())


if __name__ == "__main__":
    main()
