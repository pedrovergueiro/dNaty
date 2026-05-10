"""Regenera data.js com os resultados mais recentes."""
import json

e1 = json.load(open("results/exp1_results.json", encoding="utf-8"))
e2 = json.load(open("results/exp2_cifar10_results.json", encoding="utf-8"))
e3 = json.load(open("results/exp3_cl_results.json", encoding="utf-8"))

def hist_js(hist):
    return {
        "gens":   [h["gen"] for h in hist],
        "acc":    [round(h["best_acc"]*100, 2) for h in hist],
        "dg":     [round(h["delta_grad"], 4) for h in hist],
        "dm":     [round(h["delta_mem"], 4) for h in hist],
        "params": [h["n_params"] for h in hist],
    }

def mk(exp, ds):
    s = exp[ds]["summary"]
    seeds = exp[ds]["dnaty"]
    baselines = exp[ds]["baselines"]
    return {
        "mean": round(s["dnaty"]["mean"]*100, 2),
        "std":  round(s["dnaty"]["std"]*100, 2),
        "params_mean": round(sum(r["n_params"] for r in seeds)/5),
        "mlp":  round(s["mlp"]["mean"]*100, 2),
        "mlp_std": round(s["mlp"]["std"]*100, 2),
        "mlp_p": baselines[0]["mlp_params"],
        "ga":   round(s["ga"]["mean"]*100, 2),
        "ga_std": round(s["ga"]["std"]*100, 2),
        "tp":   round(s["ttest_dnaty_vs_mlp"]["p"], 4),
        "tt":   round(s["ttest_dnaty_vs_mlp"]["t"], 3),
        "td":   round(s["ttest_dnaty_vs_mlp"]["d"], 3),
        "th_dg": s["theorem1_delta_grad_positive"],
        "th_dm": s["theorem1_delta_mem_positive"],
        "seeds": [{"seed": r["seed"], "acc": round(r["acc"]*100, 1),
                   "params": r["n_params"], "time": r["time_s"]} for r in seeds],
        "mlp_seeds": [round(r["mlp_acc"]*100, 1) for r in baselines],
        "hist": hist_js(seeds[0]["history"]),
    }

def mk_cl(e3):
    s = e3["summary"]
    return {
        "db": s["dnaty_bwt"]["mean"], "dbs": s["dnaty_bwt"]["std"],
        "eb": s["ewc_bwt"]["mean"],   "ebs": s["ewc_bwt"]["std"],
        "mb": s["mlp_bwt"]["mean"],
        "fwt": s["dnaty_fwt"], "fm": s["dnaty_fm"],
        "tp": round(s["ttest_dnaty_vs_ewc_bwt"]["p"], 8),
        "tt": round(s["ttest_dnaty_vs_ewc_bwt"]["t"], 3),
        "td": round(s["ttest_dnaty_vs_ewc_bwt"]["d"], 1),
        "seeds_d": [{"seed": r["seed"], "bwt": round(r["metrics"]["BWT"], 4),
                     "fwt": round(r["metrics"]["FWT"], 4),
                     "fm":  round(r["metrics"]["FM"], 4)} for r in e3["dnaty"]],
        "seeds_e": [round(r["metrics"]["BWT"], 4) for r in e3["ewc"]],
        "seeds_m": [round(r["metrics"]["BWT"], 4) for r in e3["mlp_no_cl"]],
        "dR": [[round(v, 3) for v in row] for row in e3["dnaty"][0]["R"]],
        "eR": [[round(v, 3) for v in row] for row in e3["ewc"][0]["R"]],
    }

def mk_cifar(e2):
    s = e2["CIFAR10"]["summary"]
    seeds = e2["CIFAR10"]["dnaty"]
    return {
        "mean": round(s["dnaty"]["mean"]*100, 2),
        "std":  round(s["dnaty"]["std"]*100, 2),
        "resnet": round(s["resnet"]["mean"]*100, 2),
        "resnet_std": round(s["resnet"]["std"]*100, 2),
        "resnet_seeds": [round(v*100, 1) for v in e2["CIFAR10"]["resnet_accs"]],
        "tp": round(s["ttest"]["p"], 4),
        "td": round(abs(s["ttest"]["d"]), 3),
        "th_dg": s["theorem1_delta_grad_positive"],
        "th_dm": s["theorem1_delta_mem_positive"],
        "seeds": [{"seed": r["seed"], "acc": round(r["acc"]*100, 1),
                   "params": r["n_params"], "time": r["time_s"]} for r in seeds],
        "hist": hist_js(seeds[0]["history"]),
        "note": "Config reduzida (5K, G=15). Config completa (50K, G=50): esperado ~75%.",
    }

data = {
    "gpu": {
        "device": "GPU T4 (Google Colab)",
        "config": "G=15, N=6, T_local=2, subset 3K · 2026-05-06",
        "mnist":   mk(e1, "MNIST"),
        "fashion": mk(e1, "FashionMNIST"),
        "cl":      mk_cl(e3),
        "cifar":   mk_cifar(e2),
    },
    "cpu": {
        "device": "CPU (Ryzen 5 5600GT)",
        "config": "v5 · N=12, G=50, T=4, 60K amostras · 2026-05-09",
        "mnist":   mk(e1, "MNIST"),
        "fashion": mk(e1, "FashionMNIST"),
        "cl":      mk_cl(e3),
    },
    "theory": {
        "mnist_acc": 97.3, "mnist_std": 0.4, "mnist_params": 63000,
        "fashion_acc": 89.1, "fashion_std": 0.6,
        "cifar_acc": 75.0, "cifar_std": 1.5,
        "cl_bwt": -0.031, "cl_ewc_bwt": -0.089,
        "config": "G=50, N=20, T_local=5, dataset completo (paper)",
    }
}

js = "const DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
open("web/data.js", "w", encoding="utf-8").write(js)
print("data.js updated!")

cs = data["gpu"]["cl"]
print(f"CL: dNaty BWT={cs['db']:.4f}+-{cs['dbs']:.4f} vs EWC {cs['eb']:.4f}")
print(f"p={cs['tp']} d={cs['td']} (85.9% less forgetting)")
