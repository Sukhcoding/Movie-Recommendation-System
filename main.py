import json
import pickle

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the nlp model and tfidf vectorizer from disk
clf = pickle.load(open("nlp_model.pkl", "rb"))
vectorizer = pickle.load(open("tranform.pkl", "rb"))


def create_similarity():
    data = pd.read_csv("main_data.csv")

    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data["comb"])

    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity


def rcmd(m):
    global data, similarity
    m = m.lower()

    try:
        data.head()
        similarity.shape
    except Exception:
        data, similarity = create_similarity()

    if m not in data["movie_title"].unique():
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"

    i = data.loc[data["movie_title"] == m].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]  # excluding first item since it is the requested movie itself

    l = []
    for item in lst:
        a = item[0]
        l.append(data["movie_title"][a])

    return l


def convert_to_list(value):
    """
    Safely convert incoming stringified lists into Python lists.
    Handles JSON-style strings and falls back gracefully.
    """
    if isinstance(value, list):
        return value

    if value is None:
        return []

    value = value.strip()

    if value == "":
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # fallback for old custom format
    try:
        my_list = value.split('","')
        my_list[0] = my_list[0].replace('["', "").replace("['", "")
        my_list[-1] = my_list[-1].replace('"]', "").replace("']", "")
        return my_list
    except Exception:
        return []


def get_suggestions():
    data = pd.read_csv("main_data.csv")
    return list(data["movie_title"].str.capitalize())


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template("home.html", suggestions=suggestions)


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form["name"]
    rc = rcmd(movie)

    if isinstance(rc, str):
        return rc

    m_str = "---".join(rc)
    return m_str


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # getting data from AJAX request
        title = request.form["title"]
        cast_ids = request.form["cast_ids"]
        cast_names = request.form["cast_names"]
        cast_chars = request.form["cast_chars"]
        cast_bdays = request.form["cast_bdays"]
        cast_bios = request.form["cast_bios"]
        cast_places = request.form["cast_places"]
        cast_profiles = request.form["cast_profiles"]
        imdb_id = request.form["imdb_id"]
        poster = request.form["poster"]
        genres = request.form["genres"]
        overview = request.form["overview"]
        vote_average = request.form["rating"]
        vote_count = request.form["vote_count"]
        release_date = request.form["release_date"]
        runtime = request.form["runtime"]
        status = request.form["status"]
        rec_movies = request.form["rec_movies"]
        rec_posters = request.form["rec_posters"]

        # get movie suggestions for auto complete
        suggestions = get_suggestions()

        # convert stringified lists safely
        rec_movies = convert_to_list(rec_movies)
        rec_posters = convert_to_list(rec_posters)
        cast_names = convert_to_list(cast_names)
        cast_chars = convert_to_list(cast_chars)
        cast_profiles = convert_to_list(cast_profiles)
        cast_bdays = convert_to_list(cast_bdays)
        cast_bios = convert_to_list(cast_bios)
        cast_places = convert_to_list(cast_places)

        # convert cast_ids safely
        try:
            cast_ids = json.loads(cast_ids)
        except Exception:
            cast_ids = cast_ids.strip("[]").split(",")

        cast_ids = [str(x).strip().strip('"').strip("'") for x in cast_ids]

        # render escaped strings correctly
        for i in range(len(cast_bios)):
            cast_bios[i] = cast_bios[i].replace(r"\n", "\n").replace(r"\"", '"')

        # combine data for templates
        movie_cards = {
            rec_posters[i]: rec_movies[i]
            for i in range(min(len(rec_posters), len(rec_movies)))
        }

        casts = {
            cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]]
            for i in range(min(len(cast_names), len(cast_ids), len(cast_chars), len(cast_profiles)))
        }

        cast_details = {
            cast_names[i]: [
                cast_ids[i],
                cast_profiles[i],
                cast_bdays[i],
                cast_places[i],
                cast_bios[i],
            ]
            for i in range(
                min(
                    len(cast_names),
                    len(cast_ids),
                    len(cast_profiles),
                    len(cast_bdays),
                    len(cast_places),
                    len(cast_bios),
                )
            )
        }

        # default empty reviews so template still loads even if scraping fails
        movie_reviews = {}

        # web scraping to get user reviews from IMDB site
        url = f"https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/85.0.4183.83 Safari/537.36"
            )
        }

        try:
            print(f"calling imdb reviews page: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            print("IMDB status code:", response.status_code)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "lxml")
                soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})

                reviews_list = []
                reviews_status = []

                for review in soup_result:
                    review_text = review.get_text(strip=True)
                    if review_text:
                        reviews_list.append(review_text)

                        movie_review_list = np.array([review_text])
                        movie_vector = vectorizer.transform(movie_review_list)
                        pred = clf.predict(movie_vector)

                        # pred may be [0]/[1], bool, or label
                        label = pred[0] if hasattr(pred, "__len__") else pred
                        reviews_status.append("Good" if int(label) == 1 else "Bad")

                movie_reviews = {
                    reviews_list[i]: reviews_status[i]
                    for i in range(min(len(reviews_list), len(reviews_status)))
                }
            else:
                print("Failed to retrieve reviews: non-200 response")

        except Exception as e:
            print("Failed to retrieve reviews:", e)

        # Always return a response
        return render_template(
            "recommend.html",
            title=title,
            poster=poster,
            overview=overview,
            vote_average=vote_average,
            vote_count=vote_count,
            release_date=release_date,
            runtime=runtime,
            status=status,
            genres=genres,
            movie_cards=movie_cards,
            reviews=movie_reviews,
            casts=casts,
            cast_details=cast_details,
            suggestions=suggestions,
        )

    except Exception as e:
        print("Error in /recommend route:", e)
        return f"Error in recommend route: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)