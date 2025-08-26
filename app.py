import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# --- Config (must be first!) ---
st.set_page_config(page_title="🎬 Movie Intelligence", layout="wide")

# --- Load Data ---
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))
new_df = pickle.load(open('movies.pkl', 'rb'))
tmdb = pd.read_csv('tmdb_5000_movies.csv')[['id', 'title', 'budget', 'revenue', 'vote_average']]

# --- Labeling Based on Profit ---
tmdb['label'] = tmdb.apply(
    lambda row: 1 if row['revenue'] > row['budget'] and row['budget'] > 0 else 0, axis=1
)

# --- Merge DataFrames ---
df = new_df.merge(tmdb[['id', 'budget', 'revenue', 'label']], on='id')

# --- TF-IDF Processing ---
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['tags']).toarray()
y = df['label']

# --- Model Training ---
@st.cache_resource
def get_model():
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

model = get_model()

# --- Helper Functions ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=834f4f7604f1e118e0e25466da9ae622"
    data = requests.get(url).json()
    path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500{path}" if path else "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    names, posters = [], []
    for i in movie_list:
        mid = movies.iloc[i[0]].id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(mid))
    return names, posters

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 10px;
        letter-spacing: 1px;
    }
    .block-container { padding-top: 1rem; }
    .poster-title { text-align: center; font-weight: bold; font-size: 16px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='main-title'>🎬 Movie Intelligence System</div>", unsafe_allow_html=True)
st.markdown("##### Discover Similar Movies & Predict Success Based on Real-world Revenue!")

tab1, tab2 = st.tabs(["🎞️ Recommender", "📈 Success Predictor"])

# --- Tab 1: Recommender ---
# --- Tab 1: Recommender ---
with tab1:
    st.subheader("🔍 Discover Movies Like Your Favorite")
    movie_name = st.selectbox("🎥 Choose a movie:", movies['title'].values)

    if st.button("🎯 Get Recommendations"):
        names, posters = recommend(movie_name)
        st.markdown("### 🍿 Top 5 Recommended Movies")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(posters[i], use_container_width=True)  # ✅ Fixed warning here
                st.markdown(f"<div class='poster-title'>{names[i]}</div>", unsafe_allow_html=True)


# --- Tab 2: Success Classifier ---
with tab2:
    st.subheader("🎯 Predict if the Movie Was a Hit or Flop")

    selected = st.selectbox("📽️ Select a movie to classify:", df['title'].values)
    tag_text = df[df['title'] == selected]['tags'].values[0]
    vector = tfidf.transform([tag_text])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][prediction]

    movie_id = df[df['title'] == selected]['id'].values[0]
    movie_info = tmdb[tmdb['id'] == movie_id]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🔎 Prediction Result")
        st.markdown(f"**🎯 Outcome:** {'✅ Hit' if prediction == 1 else '❌ Flop'}")
        st.metric("📊 Confidence", f"{probability*100:.2f}%")
        st.metric("💵 Budget", f"${int(movie_info['budget'].values[0]):,}")
        st.metric("🎟️ Revenue", f"${int(movie_info['revenue'].values[0]):,}")

    with col2:
        st.markdown("### 🧠 Features Used")
        st.markdown(
            f"<div style='padding:10px; background-color:#f9f9f9; border-radius:6px; "
            f"font-family:monospace; font-size:14px; color:#444;'>{tag_text}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.info("✅ This prediction is based on TF-IDF of plot, cast, and genre, and trained using actual profit-based success.")
