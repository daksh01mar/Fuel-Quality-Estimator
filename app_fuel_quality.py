# app_fuel_quality.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Local file paths (your uploads) ----------
SCALER_PATH = "/mnt/data/scaler.joblib"
PLS_PATH    = "/mnt/data/pls_model.joblib"
RF_PATH     = "/mnt/data/rf_model.joblib"
TRAIN_XLSX  = "/mnt/data/diesel_properties_clean.xlsx"
SPEC_XLSX   = "/mnt/data/diesel_spec.xlsx"

# --------------- Helpers ----------------
@st.cache_resource
def load_artifacts():
    out = {}
    # scaler
    if not os.path.exists(SCALER_PATH):
        st.error(f"Missing scaler at {SCALER_PATH}")
        return None
    out['scaler'] = joblib.load(SCALER_PATH)

    out['pls'] = joblib.load(PLS_PATH) if os.path.exists(PLS_PATH) else None
    out['rf']  = joblib.load(RF_PATH)  if os.path.exists(RF_PATH) else None

    # training df (to get column names / means)
    if not os.path.exists(TRAIN_XLSX):
        st.error(f"Missing training table at {TRAIN_XLSX}")
        return None
    out['train_df'] = pd.read_excel(TRAIN_XLSX)

    # specs (min/max)
    out['spec_df'] = pd.read_excel(SPEC_XLSX) if os.path.exists(SPEC_XLSX) else None

    return out

def infer_feature_names_from_scaler(scaler):
    if hasattr(scaler, "mean_"):
        return int(len(scaler.mean_))
    if hasattr(scaler, "scale_"):
        return int(len(scaler.scale_))
    return None

def try_align_columns(train_cols, scaler):
    n = infer_feature_names_from_scaler(scaler)
    if n is None:
        return None
    if len(train_cols) == n:
        return list(train_cols)
    if hasattr(scaler, "feature_names_in_"):
        fnames = list(getattr(scaler, "feature_names_in_"))
        aligned = [f for f in fnames if f in train_cols]
        if len(aligned) == n:
            return aligned
    # fallback none
    return None

def iterative_fill_and_predict(X_init, model, scaler, known_idx, max_iter=12, tol=1e-4):
    X = X_init.copy().astype(float)
    for it in range(max_iter):
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = (X - getattr(scaler, "mean_", 0)) / getattr(scaler, "scale_", 1)
        y = model.predict(Xs)
        y = np.array(y).flatten()
        if y.shape[0] == X.shape[1]:
            # assume y in scaled space -> inverse
            try:
                cand = scaler.inverse_transform(y.reshape(1, -1)).reshape(-1)
            except Exception:
                cand = y
            for idx in known_idx:
                cand[idx] = X[0, idx]
            diff = np.nanmean(np.abs(cand - X.reshape(-1)))
            X = cand.reshape(1, -1)
            if diff < tol:
                return X.reshape(-1), y, True
        else:
            return X.reshape(-1), y, False
    return X.reshape(-1), y, False

# ---------------- Quality scoring ----------------
def build_spec_map(spec_df):
    """Return dict param -> (min,max,weight,mandatory_flag) using heuristics."""
    spec_map = {}
    if spec_df is None:
        return spec_map
    cols = list(map(str.lower, spec_df.columns))
    # try to find parameter, min, max, weight, mandatory
    # common names: 'parameter','min','max','lower','upper','weight','mandatory'
    def find_col(names):
        for n in names:
            if n in cols:
                return spec_df.columns[cols.index(n)]
        return None
    pcol = find_col(['parameter','param','property','name'])
    mincol = find_col(['min','lower','minimum'])
    maxcol = find_col(['max','upper','maximum'])
    weightcol = find_col(['weight','importance'])
    mancol = find_col(['mandatory','critical','mustpass','require'])
    if pcol and mincol and maxcol:
        for _, r in spec_df.iterrows():
            pname = str(r[pcol])
            try:
                pmin = float(r[mincol])
                pmax = float(r[maxcol])
            except:
                continue
            w = float(r[weightcol]) if weightcol and not pd.isna(r[weightcol]) else 1.0
            man = bool(r[mancol]) if mancol and not pd.isna(r[mancol]) else False
            spec_map[pname] = (pmin, pmax, w, man)
    return spec_map

def score_against_spec(predictions, names, spec_map):
    """
    predictions: dict name->value
    returns: dict name->(score, pass_bool, details)
    score ranges 0..1 (1 perfect in spec), pass_bool True if in spec or within tolerance
    """
    out = {}
    for n,v in predictions.items():
        if n in spec_map:
            pmin,pmax,w,mand = spec_map[n]
            # pass if within [min,max]
            in_spec = (v >= pmin) and (v <= pmax)
            if in_spec:
                score = 1.0
            else:
                # compute distance outside normalized by span (so further away -> worse)
                span = max((pmax - pmin), 1e-6)
                if v < pmin:
                    score = max(0.0, 1.0 - (pmin - v) / (span*2))  # degrade smoothly
                else:
                    score = max(0.0, 1.0 - (v - pmax) / (span*2))
            out[n] = {"score": float(score), "in_spec": bool(in_spec), "min":pmin, "max":pmax, "weight":float(w), "mandatory":bool(mand)}
        else:
            out[n] = {"score": None, "in_spec": None, "min":None, "max":None, "weight":1.0, "mandatory":False}
    return out

def aggregate_score(spec_scores):
    # Weighted average of defined scores; missing score -> handled with neutral 0.5 penalty
    total_w = 0.0
    acc = 0.0
    missing = 0
    mandatory_fail = False
    for n,info in spec_scores.items():
        w = info.get("weight",1.0)
        s = info.get("score", None)
        if info.get("mandatory", False) and info.get("in_spec") is False:
            mandatory_fail = True
        if s is None:
            missing += 1
            # treat missing as neutral 0.5 with normal weight
            acc += 0.5 * w
            total_w += w
        else:
            acc += s * w
            total_w += w
    if total_w == 0:
        avg = 0.0
    else:
        avg = acc / total_w
    return float(avg), bool(mandatory_fail), missing

def grade_from_score(score, mandatory_fail=False):
    if mandatory_fail:
        return "Reject (mandatory spec failed)"
    if score >= 0.9:
        return "Excellent"
    if score >= 0.75:
        return "Good"
    if score >= 0.5:
        return "Marginal"
    return "Poor / Reject"

# ---------------- GUI ----------------
st.set_page_config(page_title="Fuel Quality Estimator", layout="wide")
st.title("Fuel property predictor + fuel quality estimator")

art = load_artifacts()
if not art:
    st.stop()

scaler = art['scaler']
pls = art['pls']
rf = art['rf']
train_df = art['train_df']
spec_df = art['spec_df']

# align columns
train_cols = list(train_df.columns)
aligned_cols = try_align_columns(train_cols, scaler)
if aligned_cols is None:
    st.warning("Could not auto-align scaler/train columns. Paste exact header (CSV style) if you have it.")
    manual = st.text_area("Paste exact training header (comma-separated)", value="")
    if manual.strip():
        manual_list = [s.strip() for s in manual.split(",") if s.strip()]
        if len(manual_list) == infer_feature_names_from_scaler(scaler):
            aligned_cols = manual_list
            st.success("Using provided header.")
        else:
            st.error("Header length mismatch with scaler. Fix header or upload original training CSV.")
            st.stop()
    else:
        st.stop()

n_features = len(aligned_cols)
st.markdown(f"Using {n_features} features: (first 20 shown)")
st.write(aligned_cols[:20])

# model choice
model_choice = st.radio("Predictor model", options=[
    "PLS" if pls is not None else "PLS (missing)",
    "Random Forest" if rf is not None else "Random Forest (missing)"
])
model = pls if "PLS" in model_choice and pls is not None else rf if "Random" in model_choice and rf is not None else None
if model is None:
    st.error("No model available.")
    st.stop()

# parameter selection
col1, col2 = st.columns(2)
with col1:
    pA = st.selectbox("Parameter A", aligned_cols, index=0)
    vA = st.number_input("Value for A", value=float(train_df[pA].mean()))
with col2:
    pB = st.selectbox("Parameter B", aligned_cols, index=1)
    vB = st.number_input("Value for B", value=float(train_df[pB].mean()))

if pA == pB:
    st.warning("Choose two different parameters.")
    st.stop()

# prepare init vector (use scaler.mean_ if present)
if hasattr(scaler, "mean_"):
    init_X = scaler.mean_.reshape(1, -1).astype(float)
else:
    col_means = [float(train_df[c].mean()) if c in train_df.columns else 0.0 for c in aligned_cols]
    init_X = np.array(col_means).reshape(1, -1)

iA = aligned_cols.index(pA)
iB = aligned_cols.index(pB)
init_X[0,iA] = vA
init_X[0,iB] = vB

st.write("Running prediction...")
with st.spinner("Predicting..."):
    final_X, model_out, converged = iterative_fill_and_predict(init_X, model, scaler, known_idx=[iA, iB], max_iter=20)

pred_dict = {aligned_cols[i]: float(final_X[i]) for i in range(n_features)}
st.subheader("Predicted parameter values")
st.dataframe(pd.DataFrame.from_dict(pred_dict, orient='index', columns=['predicted_value']))

# build spec map & score
spec_map = build_spec_map(spec_df)
spec_scores = score_against_spec(pred_dict, aligned_cols, spec_map)
agg_score, mandatory_fail, missing_count = aggregate_score(spec_scores)
grade = grade_from_score(agg_score, mandatory_fail=mandatory_fail)

st.subheader("Quality estimate")
st.write(f"Aggregate score: **{agg_score:.3f}**")
st.write(f"Grade: **{grade}**")
if missing_count > 0:
    st.info(f"{missing_count} parameter(s) have no spec available and were scored heuristically (neutral).")

# show per-parameter result table
table_rows = []
for name, val in pred_dict.items():
    s = spec_scores.get(name, {})
    table_rows.append({
        "parameter": name,
        "predicted": val,
        "in_spec": s.get("in_spec"),
        "spec_min": s.get("min"),
        "spec_max": s.get("max"),
        "score": s.get("score"),
        "weight": s.get("weight"),
        "mandatory": s.get("mandatory")
    })
res_df = pd.DataFrame(table_rows)
st.dataframe(res_df)

# highlight failing mandatory items
fails = res_df[(res_df['in_spec']==False) & (res_df['mandatory']==True)]
if not fails.empty:
    st.error("Mandatory spec failures detected:")
    st.dataframe(fails)

# download
st.download_button("Download predicted parameters & quality report (CSV)",
                   res_df.to_csv(index=False),
                   file_name="fuel_quality_report.csv",
                   mime="text/csv")

st.markdown("---")
st.markdown("**Notes / Improvements you can make for a robust project deliverable:**")
st.markdown("""
- If you have a labelled quality/class column (e.g., 'quality_label' in training data), you can train a classifier (Logistic/RandomForest) to map predicted properties -> quality label; upload that model and we'll use it as the final estimator instead of heuristic scoring.
- Provide an authoritative spec file with columns (parameter, min, max, weight, mandatory) so scoring uses real pass/fail logic.
- Add parameter-specific weights (e.g., sulphur could be mandatory with high weight; cetane number may be high importance).
- Add uncertainty estimates (e.g., use ensemble predictions from RF to show confidence intervals).
""")
