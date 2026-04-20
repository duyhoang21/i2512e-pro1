import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cars Data Analysis",
    layout="wide",
    page_icon="🚗"
)

st.markdown(
    "<h1 style='text-align:center;color:#2E7D32;'>🚗 Cars Data Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- SESSION STATE ----------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("📂 Load Dataset"):
    st.session_state.raw_df = pd.read_csv("Cars_data.csv")
    st.success("Dataset loaded successfully!")

# ================= DATA SUMMARY =================
if st.session_state.raw_df is not None:
    df = st.session_state.raw_df

    st.sidebar.markdown("### 📊 Dataset Summary")
    st.sidebar.metric("Rows", df.shape[0])
    st.sidebar.metric("Columns", df.shape[1])
    st.sidebar.metric("Null Values", df.isnull().sum().sum())

# ================= RAW DATA =================
if st.session_state.raw_df is not None:
    df = st.session_state.raw_df.copy()

    st.subheader("🔹 Raw Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # ---------- BEFORE PREPROCESSING ----------
    st.subheader("🧪 Before Preprocessing Analysis")

    before_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Null Count": df.isnull().sum().values
    })
    st.dataframe(before_info, use_container_width=True)

    # ---------- NULL VISUAL ----------
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Column", "Null Count"]

    st.plotly_chart(
        px.bar(
            null_df.sort_values("Null Count"),
            x="Null Count",
            y="Column",
            orientation="h",
            title="Missing Values per Column (Before)"
        ),
        use_container_width=True
    )

    # ================= BEFORE VISUALS =================
    st.subheader("📊 Visualizations Before Preprocessing")

    if "Ex-Showroom_Price" in df.columns:
        st.plotly_chart(
            px.histogram(
                df,
                x="Ex-Showroom_Price",
                nbins=40,
                title="Ex-Showroom Price Distribution (Before)"
            ),
            use_container_width=True
        )

    if "Make" in df.columns:
        make_count = df["Make"].value_counts().head(10).reset_index()
        make_count.columns = ["Make", "Count"]

        st.plotly_chart(
            px.bar(
                make_count,
                x="Count",
                y="Make",
                orientation="h",
                title="Top 10 Car Brands (Before)"
            ),
            use_container_width=True
        )

    if "Fuel_Type" in df.columns:
        st.plotly_chart(
            px.pie(
                df,
                names="Fuel_Type",
                title="Fuel Type Distribution (Before)"
            ),
            use_container_width=True
        )

    if {"Seating_Capacity", "Ex-Showroom_Price"}.issubset(df.columns):
        st.plotly_chart(
            px.box(
                df,
                x="Seating_Capacity",
                y="Ex-Showroom_Price",
                title="Price vs Seating Capacity (Before)"
            ),
            use_container_width=True
        )

# ================= PREPROCESSING =================
if st.session_state.raw_df is not None:
    if st.sidebar.button("🧹 Start Preprocessing"):
        df = st.session_state.raw_df.copy()

        df["Ex-Showroom_Price"] = (
            df["Ex-Showroom_Price"]
            .astype(str)
            .str.replace("Rs.", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Ex-Showroom_Price"] = pd.to_numeric(df["Ex-Showroom_Price"], errors="coerce")

        def extract_numeric(series):
            return series.astype(str).str.extract(r'([\d\.]+)')[0].astype(float)

        for col in ["Displacement", "Power", "Torque", "Mileage"]:
            if col in df.columns:
                df[col + "_num"] = extract_numeric(df[col])

        if {"Power_num", "Displacement_num"}.issubset(df.columns):
            df["Engine_Efficiency"] = df["Power_num"] / df["Displacement_num"]

        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

        df.drop_duplicates(inplace=True)

        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        scaler = MinMaxScaler()
        clean_df = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns
        )

        st.session_state.clean_df = clean_df
        st.success("✅ Preprocessing Completed Successfully!")

# ================= AFTER PREPROCESSING =================
if st.session_state.clean_df is not None:
    clean_df = st.session_state.clean_df

    st.markdown("---")
    st.subheader("✅ After Preprocessing Analysis")

    # ✅ ADDED: AFTER PREPROCESSING SUMMARY TABLE
    after_info = pd.DataFrame({
        "Column": clean_df.columns,
        "Data Type": clean_df.dtypes.astype(str),
        "Null Count": clean_df.isnull().sum().values
    })
    st.dataframe(after_info, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    a1.metric("Rows", clean_df.shape[0])
    a2.metric("Columns", clean_df.shape[1])
    a3.metric("Null Values", clean_df.isnull().sum().sum())

    st.subheader("📊 Cleaned Dataset Preview")
    st.dataframe(clean_df, use_container_width=True)

    # ================= AFTER VISUALS =================
    st.subheader("📊 Visualizations After Preprocessing")

    if "Ex-Showroom_Price" in clean_df.columns:
        st.plotly_chart(
            px.histogram(
                clean_df,
                x="Ex-Showroom_Price",
                title="Scaled Price Distribution (After)"
            ),
            use_container_width=True
        )

    if {"Engine_Efficiency", "Ex-Showroom_Price"}.issubset(clean_df.columns):
        st.plotly_chart(
            px.scatter(
                clean_df,
                x="Engine_Efficiency",
                y="Ex-Showroom_Price",
                opacity=0.6,
                title="Engine Efficiency vs Price (After)"
            ),
            use_container_width=True
        )

    if "Seating_Capacity" in clean_df.columns:
        st.plotly_chart(
            px.violin(
                clean_df,
                y="Seating_Capacity",
                box=True,
                title="Seating Capacity Distribution (After)"
            ),
            use_container_width=True
        )

    important_cols = [
        col for col in [
            "Ex-Showroom_Price",
            "Power_num",
            "Displacement_num",
            "Mileage_num",
            "Engine_Efficiency",
            "Seating_Capacity"
        ]
        if col in clean_df.columns
    ]

    if len(important_cols) > 2:
        st.plotly_chart(
            px.imshow(
                clean_df[important_cols].corr(),
                title="Correlation Heatmap – Key Car Features"
            ),
            use_container_width=True
        )

    st.download_button(
        "💾 Download Cleaned Cars Dataset",
        clean_df.to_csv(index=False).encode("utf-8"),
        "cleaned_cars_data.csv",
        "text/csv"
    )

