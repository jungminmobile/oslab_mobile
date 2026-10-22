"""run_e13_1.py — E13-1 하위층 빼기 사전 확인 (FP32) 드라이버. 계획서 6·8절.

  --step 0       다른 측정 확인, B/32 가중치 받기, 256 dev 캐시 생성, 분류기, G0(FP32 dev 전체 89.96±0.05), env.json
  --step smoke   후보 1개 × 2장
  --step score   후보 18개 하나씩 × dev 6×200 → score.jsonl (+ 논문식 1−cos 기록)
  --step cumul   cos 순위 상위 1·2·3개 누적 P1·P2·P3 × dev 6×200 → cumul.jsonl
  --step gate    스크리닝 고화질 Δ ≥ −0.2k 인 P_k만 dev 전체 + 층화 짝 부트스트랩 → gate.jsonl
  --step summary 계획서 5절 판정 → summary.json, REPORT.md
결과: results/E13-1_하위층빼기_사전확인/results/ , 실행 명령은 run_log.txt
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
STEP = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--step"), None)
os.environ["HF_HUB_OFFLINE"] = "0" if STEP == "0" else "1"     # 가중치 받기는 0단계에서만 (계획서 7절)
import prune_screen as E      # noqa: E402
import numpy as np            # noqa: E402

G0_REF = {"cifar_A": 75.2, "pets_dev": 92.02, "flowers_dev": 86.42, "cars_dev": 91.62, "food_dev": 85.99,
          "caltech_dev": 93.75, "hr": 89.96}                              # G8 b32 README B2
OUT = E.OUT


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def other_python():
    import psutil
    me = os.getpid()
    return [{"pid": p.pid, "cmd": " ".join(p.info["cmdline"] or [])[:200]} for p in psutil.process_iter(["name", "cmdline"])
            if p.pid != me and (p.info["name"] or "").lower().startswith("python")
            and "run_e13_1.py" not in " ".join(p.info["cmdline"] or [])]


def read_jsonl(name):
    p = OUT / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


# ---------------------------------------------------------------- 0

def step0(a):
    others = other_python()
    print("다른 python:", others, flush=True)
    sys.path.insert(0, str(E.G8))
    from scripts import env_info
    env = env_info.collect()
    env["e13"] = {**E.git_state(), "torch_threads": E.torch.get_num_threads(), "art": str(E.ART),
                  "other_python": others, "platform": platform.platform()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    t = time.time()
    model, tok = E.load()          # HF 캐시에 없으면 받음
    nparam = sum(p.numel() for p in model.visual.parameters())
    print(f"모델 로드 {time.time() - t:.0f}s, 이미지 파라미터 {nparam / 1e6:.2f} M", flush=True)
    if not (E.dev_dir() / "SHA1SUMS.json").exists() or a.rebuild:
        t = time.time()
        info, sums = E.build_dev()
        E.append("prep.jsonl", {"what": "dev256", "n": info, "sha1": sums, "sec": round(time.time() - t),
                                "preprocess": "PIL RGB → BICUBIC 256×256 바로 맞춤 → uint8 (x/127.5 − 1 은 평가 때)"})
    if not (E.ART / "clf" / "caltech_dev_clf.npy").exists() or a.rebuild:
        t = time.time()
        E.build_clf(model, tok)
        E.append("prep.jsonl", {"what": "clf", "sec": round(time.time() - t), "templates": "CIFAR RC.TEMPLATES, 고화질 RD.HIGHRES 템플릿 (CLIP)"})
    # G0: 전체 모델 FP32 dev 전체 → ref 임베딩
    (E.ART / "ref").mkdir(exist_ok=True)
    res, t0 = {}, time.time()
    for key in E.DEV:
        pix, y, W = E.load_dev(key)
        t = time.time()
        e = E.embed(model, pix)
        np.save(E.ref_path(key), e)
        res[key] = {"n": int(len(y)), "top1": round(100 * float(((e @ W).argmax(1) == y).mean()), 2),
                    "ms_per_img": round(1000 * (time.time() - t) / len(y), 1)}
        print(key, res[key], flush=True)
    hr = float(np.mean([res[k]["top1"] for k in E.HRK]))
    ok = abs(hr - G0_REF["hr"]) <= 0.05
    E.append("g0.jsonl", {"what": "G0", "per_set": res, "hr_mean": round(hr, 3), "ref": G0_REF, "pass": ok,
                          "image_params_M": round(nparam / 1e6, 3), "sec": round(time.time() - t0), "macs": E.macs()})
    if not ok:
        raise SystemExit(f"G0 실패: 고화질 평균 {hr:.3f} (기준 89.96±0.05) — 멈춤")


# ---------------------------------------------------------------- smoke / score / cumul / gate

def smoke(a):
    model, _ = E.load()
    sub = E.CANDIDATES[0]
    pix, y, W = E.load_dev("pets_dev", 2)
    ref = np.load(E.ref_path("pets_dev"))[:2]
    e0 = E.embed(model, pix)
    with E.Skip(model, [sub]):
        e1 = E.embed(model, pix)
    e2 = E.embed(model, pix)                   # hook 해제 후 원래대로
    with E.SubCos(model) as sc:
        E.embed(model, pix)
    r = {"what": "smoke", "sub": E.name(sub), "cos_full_vs_ref": (e0 * ref).sum(1).tolist(),
         "cos_skip_vs_ref": (e1 * ref).sum(1).tolist(), "restore_maxdiff": float(np.abs(e2 - e0).max()),
         "subcos_keys": len(sc.result()), "params": E.sub_params(model, sub), "share": E.share([sub])}
    E.append("smoke.jsonl", r)
    assert min(r["cos_full_vs_ref"]) > 0.99999 and r["restore_maxdiff"] < 1e-5 and max(r["cos_skip_vs_ref"]) < 1


def row_of(model, subs, res, hr):
    return {"subs": [E.name(s) for s in subs], "share_exact_pct": E.share(subs), "share_plan_pct": E.share(subs, False),
            "params_removed": int(sum(E.sub_params(model, s) for s in subs)),
            "hr_cos": hr["cos"], "all6_cos": float(np.mean([res[k]["cos"] for k in E.DEV])),
            "hr_agree": hr["agree"], "all6_agree": float(np.mean([res[k]["agree"] for k in E.DEV])),
            "hr_dtop1": hr["dtop1"], "hr_top1": hr["top1"], "cifar_dtop1": res["cifar_A"]["dtop1"], "per_set": res}


def score(a):
    model, _ = E.load()
    t0 = time.time()
    with E.SubCos(model) as sc:            # 기준 forward (전체 모델) + 논문식 점수
        res0, hr0, _ = E.evaluate(model, [], E.SCREEN_N)
    E.append("score.jsonl", {"what": "base", **row_of(model, [], res0, hr0), "paper_1mcos": sc.result(),
                             "sec": round(time.time() - t0)})
    paper = sc.result()
    for i, sub in enumerate(E.CANDIDATES):
        t = time.time()
        res, hr, _ = E.evaluate(model, [sub], E.SCREEN_N)
        E.append("score.jsonl", {"what": "single", **row_of(model, [sub], res, hr), "paper_1mcos": paper[E.name(sub)],
                                 "sec": round(time.time() - t)})
        print(f"--- {i + 1}/18 {E.name(sub)} hr_cos {hr['cos']:.4f} Δ {hr['dtop1']:+.2f}", flush=True)


def ranking():
    rows = [r for r in read_jsonl("score.jsonl") if r["what"] == "single"]
    latest = {}
    for r in rows:
        latest[r["subs"][0]] = r
    assert len(latest) == 18, len(latest)
    return sorted(latest.values(), key=lambda r: -r["hr_cos"])     # 고화질 cos 평균 높은(영향 작은) 순


def parse(n):
    k = "attn" if n.startswith("attn") else "mlp"
    return (k, int(n[len(k):]))


def cumul(a):
    model, _ = E.load()
    rank = ranking()
    for k in (1, 2, 3):
        subs = [parse(r["subs"][0]) for r in rank[:k]]
        t = time.time()
        res, hr, _ = E.evaluate(model, subs, E.SCREEN_N)
        E.append("cumul.jsonl", {"what": f"P{k}", "k": k, **row_of(model, subs, res, hr), "sec": round(time.time() - t)})


def gate(a):
    sys.path.insert(0, str(E.G8))
    from s2_paired_ci import boot_hr
    model, _ = E.load()
    rows = {r["what"]: r for r in read_jsonl("cumul.jsonl")}
    for k in (1, 2, 3):
        r = rows[f"P{k}"]
        if r["hr_dtop1"] < -0.2 * k:
            print(f"P{k}: 스크리닝 Δ {r['hr_dtop1']:+.2f} < {-0.2 * k:.1f} → 관문 측정 안 함", flush=True)
            continue
        subs = [parse(n) for n in r["subs"]]
        t = time.time()
        res, hr, kp = E.evaluate(model, subs, None, keep=True)
        diff = {key: kp[key]["correct"].astype(float) - kp[key]["correct_ref"] for key in E.HRK}
        ci = boot_hr(diff, 1000, 0)
        np.savez_compressed(OUT / f"pred_P{k}.npz", **{f"{key}_{c}": v for key in E.DEV for c, v in kp[key].items()})
        E.append("gate.jsonl", {"what": f"P{k}", "k": k, **row_of(model, subs, res, hr), "hr_dtop1_ci95": ci,
                                "n_boot": 1000, "seed": 0, "sec": round(time.time() - t)})


# ---------------------------------------------------------------- summary

def summary(a):
    cum = {r["what"]: r for r in read_jsonl("cumul.jsonl")}
    gat = {r["what"]: r for r in read_jsonl("gate.jsonl")}
    dec = []
    for k in (1, 2, 3):
        c, g = cum[f"P{k}"], gat.get(f"P{k}")
        d = {"P": f"P{k}", "subs": c["subs"], "share_exact_pct": c["share_exact_pct"], "share_plan_pct": c["share_plan_pct"],
             "screen_hr_dtop1": round(c["hr_dtop1"], 2), "screen_pass": c["hr_dtop1"] >= -0.2 * k}
        if g:
            d.update(full_hr_dtop1=round(g["hr_dtop1"], 2), ci95=g["hr_dtop1_ci95"], cifar_dtop1=round(g["cifar_dtop1"], 2))
            d["pass"] = bool(g["hr_dtop1"] >= -0.2 * k and g["hr_dtop1_ci95"][0] >= -0.5 * k and c["share_plan_pct"] >= 5)
        else:
            d["pass"] = False
        dec.append(d)
    passed = [d for d in dec if d["pass"]]
    best = max(passed, key=lambda d: (d["share_plan_pct"], d["full_hr_dtop1"])) if passed else None
    s = {"decision": f"E13-2 진행: {best['P']}" if best else "중단", "configs": dec,
         "rule": "Δ ≥ −0.2k (점추정) AND CI 하한 ≥ −0.5k AND 연산 몫 ≥ 5% (계획서 5절)", **E.git_state()}
    (OUT / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["0", "smoke", "score", "cumul", "gate", "summary"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    log(f"START {' '.join(sys.argv)} git={E.git_state()}")
    t = time.time()
    {"0": step0, "smoke": smoke, "score": score, "cumul": cumul, "gate": gate, "summary": summary}[a.step](a)
    log(f"END --step {a.step} {time.time() - t:.0f}s")
    print(f"[{a.step}] {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
