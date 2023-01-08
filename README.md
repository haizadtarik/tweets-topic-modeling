# Topic modeling of replies and quote tweets

This repo analyze the replies and quote tweets of a tweet based on the provided tweet ID and perform topic modeling using Latent Dirichlet Allocation (LDA)

The replies and quote tweets are retrived from twitter using Twitter API so twitter developer access are required to run the code.

## Setup

1. Git clone the repo
    ```
    git clone https://github.com/haizadtarik/tweets-topic-modeling.git
    ```

2. Install Dependencies
    ```
    pip install -r requirements.txt
    ```

3. Create `.env` file and specify BEARER_TOKEN
    ```
    BEARER_TOKEN = <BEARER_TOKEN_FROM_TWITTER_DEVELOPER_DASHBOARD>
    ```

## Run

1. Run the anlyze script
    ```
    python analyze.py --id <TWEET_ID> --params <PARAM_1> <PARAM_2> ... <PARAM_N>
    ```

2. Result can be viewed by opening the generated html file in web browser
