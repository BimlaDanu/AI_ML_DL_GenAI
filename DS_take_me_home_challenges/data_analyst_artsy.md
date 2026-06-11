### Artsy Data Analyst Exercise

*Note: We recommend spending no more than four hours to complete this
exercise.*

In this exercise, we'll analyze two datasets: pageviews and users.
Pageviews are what they sound like, single page loads on [https://www.artsy.net](https://www.artsy.net).

Columns include a user ID (*anonymous_id*), whether the user is logged in
(*logged_in*), a timestamp for the event (*received_at*), the url path of the pageview (*path*), and the referring path (*referrer*).


Users contains user-level data telling us how a user has interacted with Artsy thus far. It includes pageviews, which we defined above, as well as inquiries. Inquiries are an important commercial metric for Artsy’s business, representing a visitor contacting a gallery about an artwork (which may eventually lead to a purchase).

The Users dataset has the following columns: a user ID ( *visitor_id* ), whether the user has ever inquired on the site ( *has_inquired* ), whether the user has registered for an account ( *has_an_account* ), and how many *article_pageviews* , *artist_pageviews*, and *artwork_pageviews* they have had on the site.


These data sets are actually samples and may not represent true distributions.
They are nonetheless instructive and should be treated as complete for the
purpose of this exercise.

#### Data:
- artsy data test pageviews
- artsy data test user

In the following questions, we're looking for clear answers that demonstrate how
you would tackle the problem at hand. Please include supporting code (you may
link to a gist or attach a file).


### Questions:

#### Pageviews exploration
1. What was the most seen Artsy page in this dataset?
2. Artsy has thousands of unique urls for artworks, articles, and other entities on
its site. Can you think of any way to categorize these pages into "types"?
Categorize the most common ones.
3. Please provide code that would add a new column to a dataframe of
pageviews in Python (pandas) or R containing the page type you’ve defined.
4. What is the breakdown of pageviews of each page type (as you’ve defined
them) for logged in visitors? For logged out? What are some observations you
can make about users' behavior on Artsy?

#### Users analysis
Predicting which users will submit inquiries on our website is an important
business insight. We provide a dataset of users, some have inquired
(has_inquired=True) and some haven't (has_inquired=False).
1. What observations can you make about inquirers and non-inquirers's behavior
based on this dataset?
2. Using the Users dataset, build a machine learning model to predict who will
inquire.
3. Evaluate the performance of your model.
4. What other features would you want to add to the dataset to build a stronger
model?
Ԝո What are some recommendations you can make to improve our product for
commercial users based on the model that you built?