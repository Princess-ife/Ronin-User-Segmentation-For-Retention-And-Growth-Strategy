<h1>Ronin User Segmentation For Retention And Growth Strategy</h1>

<p>I built an unsupervised user Segmentation using different clustering algorithms for the Ronin blockchain to help them understand their user base to make better decisions before launching or investing in any games.</p>


<h1>Problem Statement</h1>

<p>Here are some of the problems that Ronin has, for which the clustering algorithms provide a perfect solution.</p>

<h2>Heavy dependency on a few hit games</h2>

<p>In a recent article, I discovered that Ronin's on-chain activity increased significantly when Pixel migrated from Polygon to the chain in 2024. 

However, the moment Pixel’s popularity declined, it negatively impacted Ronin’s on-chain activity, causing it to drop to 70%. 

So, it is safe to say that the chain rides on hot games, and with k-means clustering, we’ll be able to segment users by their game preferences and engagement patterns to help identify which users are "true believers" vs "tourists" following a single game.</p>
 
<a>https://cointelegraph.com/news/ronin-zksync-onchain-metrics-fell-most-2025</a>

<h2>Understanding the different types of gamers and their behaviour</h2>

<p>We all know that not all gamers are the same, even if they play the same game. Each player has different: 

» Spending habits.

» Game interests.


» Gaming session length (time frame).

» Engagement consistency. 

So, K-means clustering would help discover these partners and enable Ronin to gain certain information that will enable them to scale better on-chain.</p>


<h2>Identify whale vs. casual user segments</h2>

<p>This is linked with problem 2. Because after you have understood the player patterns, you can cluster them into segments based on their behaviours.</p>

<h2>Detecting potential churners before they leave</h2>

<p>Got this idea from the churn model Boss Joshua built. This feature would help detect pre-churners, complementing the churn risk model by identifying a group of users who are likely to churn.</p>


<h1>DATASETS</h1>

<p>I got the 7000 Ronin datasets needed for the clustering model(s) from Dune. I wrote the query and did feature engineering to get the data needed to build my model. 

Below are the new features (columns) I identified while retrieving the data on Dune.


→ Transaction count.

→ Unique address.

→ Days active.

→ Days since last transaction.

→ Wallet age days.

→ Value sent and average value sent.

After getting the data on this feature, I fetched the data using the Dune API in a Python script, and then I proceeded to download the datasets in a .csv file to begin my project.</p>


<h1>DATA EDA AND PREPROCESSING </h1>

<p>I loaded the file and literally read through it to understand everything going on in the data before using it for my model. 
  
In the process of understanding my data, I visualized the data (uncleaned) to check for outliers.</p>


<h3>Box plot</h3>

<p>I used the box plot to check for outliers (before cleaning), and features like the following:

→ Transaction count showed an outrageous outlier. It goes from over 100 to 312k.

→ Value sent also showed extreme whales with over 62m RON.

→ The unique address feature also showed that some users interacted with over 10k+ addresses. (quite ridiculous).

While other features like…

→ days active 

→ days since last transactions

→ wallet age days 

They are all bound by the window that I set in the query.</p>



<h3>Scatter plot</h3>

</p>I also did scatter plots to visualize the relationships between 2 continuous variables. I visualized:


→ Transaction count vs value sent.

→  Transaction count vs average value sent.


And the visualization showed clusters around 0 and 0.5, which indicates that some of the users are normal users because their transaction counts are below 10,000, with under $500 sent.


It also showed extreme outliers where some users performed over 30,000+transactions with low amounts. 


Another showed that some users had over $30,000 value sent with relatively low transactions.


It is safe to say…


High transactions don’t equal high spending. </p>






<h3>Handling Outliers</h3>
<p>There are numerous ways to handle outliers, which include winsorization, log transformation, removing outliers, and so on. 

But it mostly depends on the data that’s being used. Using blockchain data, every data point tells a story, so most times it isn’t advisable to remove outliers, as you could miss out on the behaviour you need for your model or analysis.

I handled the outlier here by removing the top 1% outliers.

The top 1% (3.67% to be precise) includes the outrageous outliers that looked more like a contract or bot involvement. So, when I removed outliers, the data was reduced to 6743 rows from 7000 rows, and I removed 257 rows.</p>

<h3>DATA SCALING</h3>

<p>I scaled the data using StandardScaler, as it is best for K-Means, preserves the shape of the distribution better, and works well with the centroid-based approach.</p>
  
<h1>DATA VISUALIZATION AFTER CLEANING</h1>

<p>It is important to know if there were any changes in the way the data looked or would be read.</p>

<h3>Scatter plot </h3>

<p>Even after cleaning, the plot still showed that the data is highly skewed because most of the users have low transactions and low average values. 

It also showed that it is skewed because most users are moderate, while a few are heavy users (extremely so).

To make it easier to read, I scaled the data using log.

Note: I used the scaled data here.</p>






<h3>Histogram</h3>

<p>Using the scaled data, I analyzed each feature with a histogram. And it showed that the data is mostly centered around 0 (which indicates normal users), and shows how ready the data is for training the model. 

Although some features like value sent and transaction count are highly skewed, it is normal for blockchain data. 

It is what K-means can handle.</p>



<h3>Correlation heatmap</h3>

<p>I did the correlation heatmap to show the relationship between each feature. The heatmap showed:

→Strong positive correlations between value sent and average value sent and unique address and wallet age days.

→Strong negative correlations between days active and days since last transactions, days active and wallet age days, and days active and average value sent.

→Weak or no correlations in transaction count and average gas used.


This means that most features are independent, which show different aspects of user behaviour. </p>



<h1>Elbow Point (k)</h1>

<p>The elbow point shows how spread the clusters are and how many groups we are dividing the wallets into. 
  
  After plotting, the point fell on 5, which is where the curve began to flatten.
  
But I needed to confirm, so I checked the silhouette score, which answered the question “Do wallets in the same cluster actually belong together?”

The score was 0.6028031174416356, which fell on 4. 


While the silhouette score peaked at k=4, k=5 was chosen to allow finer separation between high-value and high-activity users, which is more actionable for retention and growth strategies.</p>


<h2>PCA</h2>

<p>The PCA visualization showed all 5 clusters in the plot. In the plot, the green cluster is the largest, spread across the right side, which is cluster 1.
  
The blue cluster is cluster 0, the purple cluster is cluster 2, the teal cluster is cluster 3, and the yellow cluster is cluster 4.

The PCA literally just proved that k = 5 works for Ronin user segmentation.</p>



<h1>Cluster Meaning</h1>

<h2>Cluster 0 - Active low spenders</h2>

<p>Size: 709 wallets (10.5%)


→ have moderate activity on-chain with 343 transactions on average. 

→ They are very engaged within the 30-day window. The data showed that they are active 25.5 days, which is 85% in the 30-day window.

→ They are low spenders as their average value per transaction is only 5.59 RON.

→ Their wallet age is only 5.1 days, which means they are relatively new.

→ They have only interacted with 18 unique addresses.

It is safe to call them Enthusiasts Newbies because they are very engaged but haven’t spent money yet.


Connection to Problems:

Churn Risk (Problem 4): LOW RISK - Excellent engagement (85% of days) and very recent activity.</p>


<h2>Cluster 1- True Whales (High-Value Players)</h2>

<p>Size: 212 wallets (3.1%)

→ Transaction activity: 377 transactions average (median: 221).

→ Engagement level: Active 23.6 days out of 30 (79% engagement).

→ Recency: Moderate recent activity.

Spending behavior

→  Total value sent: 13,042 RON average (median: 11,184 RON).

→ Average per transaction: 52.11 RON (median: 43.17 RON).

→ Network interaction: 25 unique addresses (above average).

→ Gas usage is high due to complex transactions (NFTs, DeFi).
  
Connection to Problems:

Game Dependency (LOWER RISK) - Interact with 25 addresses, suggesting multi-game engagement</p>

<h2>Cluster 2 - Bots or Automated Systems like contracts (Power Traders)</h2>

<p>Size: 87-135 wallets (1.3-2%)
  
→Transaction activity: 768 transactions average (median: 251) - VERY HIGH
  
→ Engagement level: Only 7.5 days active (25%).

→ Recency: Variable

→ Spending behavior:
  
Total value sent: 321 RON average
  
Average per transaction: 0.91 RON (very small)
  
→ Network interaction: 196 unique addresses (10x normal!) - highest of all clusters
  
→ Gas usage: They are both simple and complex transactions
  
Behavioral Profile: 
  
These wallets exhibit automated or bot-like behavior rather than human gaming patterns. High transaction volume concentrated in few days, combined with interaction across 196+ unique addresses, suggests algorithmic trading, arbitrage bots, or automated market-making activities. 

  Connection to Business Problems:
  
Game Dependency (Problem 1):  NO RISK - Highly diversified across many contracts.</p>

<h2>Cluster 3: Moderate Spenders </h2>

<p>Size: 718 wallets (10.7%)
  
→ Transaction activity: 330 transactions average (median: 166)
  
→ Engagement level: 12.3 days active (41% engagement)
  
→ Spending behavior:
  
Total value sent: 1,839 RON average (median: 1,298 RON)
  
Average per transaction: 10.59 RON (median: 7.74 RON)
  
→ Network interaction: 11 unique addresses
  
→ Wallet maturity: Mix of new and established wallets
  
Behavioral Profile: 

These clusters sit between casual players and whales.

They're paying users who spend meaningfully but not at whale levels. They show moderate engagement and consistent spending patterns. Likely playing games seriously enough to invest, but managing budgets carefully.

Connection to Project Problems:

→  Game Dependency (Problem 1): MODERATE RISK - 11 addresses suggest 2-3 main games.

→  Churn Risk (Problem 4):  MODERATE - 41% engagement could slip either way</p>

<h2>Cluster 4: F2P Grinders (Daily Free Players)</h2>

<p>Size: 2,586 wallets (38.4%)
  
→ Transaction activity: 152 transactions average (median: 121)
  
→ Engagement level: 29.6 days active (99% engagement!) - plays almost daily!
  
→ Spending behavior:

Total value sent: 17 RON average (nearly zero)
  
Average per transaction: 0.07 RON (essentially free)

→ Network interaction: Only 3 unique addresses (1-2 games maximum)

→ Wallet maturity: Established users

  Behavioral Profile: 
  
These are dedicated free-to-play (F2P) grinders who play daily but refuse to spend. Highest engagement rate of any cluster (99%!), but lowest spending. They're committed to specific games but play entirely free. Represent the largest single segment at 38% of all users.

  Connection to Project Problems:
  
→  Game Dependency (Problem 1): HIGHEST RISK - Only 3 addresses = 1 game focus, extremely vulnerable.
  
→  Churn Risk (Problem 4):  PARADOX - High engagement BUT if their one game fails, massive churn.</p>



















