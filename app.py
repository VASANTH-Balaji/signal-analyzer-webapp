"""
app.py - Streamlit web frontend for the SIH26147 signal analyzer.

This is a drop-in replacement for main_gui.py (PyQt5). It calls the exact
same backend (analysis_pipeline.run_pipeline) with zero changes to any
signal-processing file. Deploy this + the backend .py files to Streamlit
Community Cloud (or any host that runs Python) to get a public URL.
"""
import io
import json
import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

import analysis_pipeline as ap

st.set_page_config(page_title="Signal Analyzer (SIH26147)", layout="wide")
st.title("Signal Analyzer")
st.caption("Upload a .iq or .wav capture. Identifies modulation, FEC, interleaving, "
           "demodulates, de-interleaves, error-corrects and correlates the bitstream.")

with st.sidebar:
    st.header("Options")
    sample_rate = st.number_input("Sample rate override (Hz, 0 = auto)", min_value=0.0, value=0.0, step=1000.0)
    modulation = st.selectbox("Modulation override", ["Auto-detect", "FSK", "PSK/BPSK", "QAM"])
    use_meta = st.checkbox("Use sidecar *_meta.json if present", value=True)
    blind_fec = st.checkbox("Blind FEC / interleaving detection", value=True)
    iq_format = st.selectbox("Sample format override", ["Auto-sniff", "float32", "float64", "int16", "uint8", "int8"])

uploaded = st.file_uploader("Signal file", type=["iq", "wav"])

# Optional sidecar metadata (only meaningful for .iq files with a matching *_meta.json)
meta_file = st.file_uploader("Optional sidecar metadata (*_meta.json)", type=["json"])

run = st.button("Analyze", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sig_path = os.path.join(tmpdir, uploaded.name)
        with open(sig_path, "wb") as f:
            f.write(uploaded.getbuffer())
        if meta_file is not None:
            meta_path = os.path.splitext(sig_path)[0] + "_meta.json"
            with open(meta_path, "wb") as f:
                f.write(meta_file.getbuffer())

        opts = ap.AnalysisOptions(
            sample_rate=sample_rate or None,
            modulation=None if modulation == "Auto-detect" else modulation,
            use_meta=use_meta,
            blind_fec=blind_fec,
            iq_format=None if iq_format == "Auto-sniff" else iq_format,
        )

        plot_data = {}
        ctx = ap.PipelineContext(on_plots=lambda d: plot_data.update(d or {}))

        with st.spinner("Running pipeline..."):
            try:
                status = ap.run_pipeline(sig_path, ctx, opts)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                status = "error"

        st.subheader(f"Status: {status}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Parameters")
            for group, rows in ctx.params.items():
                with st.expander(group, expanded=True):
                    st.table({"value": rows})

            if ctx.output:
                st.markdown("### Decoded output")
                st.write(f"{ctx.output['label']}  ({len(ctx.output['data'])} bytes)")
                st.code(bytes(ctx.output["data"][:512]).hex(), language=None)
                st.download_button("Download decoded bytes", bytes(ctx.output["data"]),
                                    file_name="decoded_output.bin")

            rep = ap.build_report(sig_path, opts, ctx)
            st.download_button("Download full report (JSON)",
                                json.dumps(rep, indent=2, default=str),
                                file_name="report.json")
            st.download_button("Download full report (text)",
                                ap.report_to_text(rep),
                                file_name="report.txt")

        with col2:
            st.markdown("### Plots")
            fig = plt.figure(figsize=(9, 7))
            ap.render_plots(fig, plot_data)
            st.pyplot(fig)

        with st.expander("Log"):
            st.text("\n".join(ctx.log_lines))
else:
    st.info("Upload a .iq or .wav file, then click Analyze.")
