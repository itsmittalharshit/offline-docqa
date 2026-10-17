import os

import requests
import streamlit as st

API_URL = os.environ.get("DOCQA_API", "http://localhost:8000")
MARKERS = {"verified": "✅", "weak": "⚠️", "unsupported": "❌"}

st.set_page_config(page_title="Offline DocQA", page_icon="📄")
st.title("Offline DocQA")
st.caption("Answers from internal reports with citations and claim checks. Runs fully offline.")

question = st.text_input("Your question (English or Hindi)")
verify = st.checkbox("Check claims against the cited sources", value=True)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the documents and writing an answer..."):
        try:
            resp = requests.post(f"{API_URL}/ask",
                                 json={"question": question, "verify": verify}, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Could not get an answer from the API: {e}")
            st.stop()

    data = resp.json()
    if data["refused"]:
        st.warning(data["answer"])
    else:
        st.markdown(data["answer"])

        if data["sources"]:
            st.subheader("Sources")
            for s in data["sources"]:
                pages = (f"p.{s['page_start']}" if s["page_start"] == s["page_end"]
                         else f"pp.{s['page_start']}–{s['page_end']}")
                st.write(f"[{s['n']}] {s['doc']}, {pages}")

        if data["invalid_citations"]:
            st.warning(f"The answer cites passages that don't exist: {data['invalid_citations']}")

        if data["claims"]:
            st.subheader("Claim check")
            for c in data["claims"]:
                st.write(f"{MARKERS[c['level']]} **{c['score']:.2f}** — {c['claim']}")

    st.caption(f"retrieval {data['retrieval_s']:.1f} s · generation {data['generation_s']:.1f} s")