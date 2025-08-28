1. use OneMAP api to draw distance between point and the nearest amenities to the location as features for price prediction
    - use more features like remaining lease which are all predictive features in predicting price
    - tried to implement nearest mrt and public transport but api rate limited and it takes too long

2. under the resale there is a column remaining lease, i think can potentially add to the other columns to match but will leave out for now

3. clustering might help more because of the way resale prices tend to be established

4. current assumption is that the additional info regarding nearest public transport is true regardless of the housing transaction at that time (eg. assume Woodlands MRT existed since 2008, though it might not be true)

5. another assumption is the recency of the data, one model is trained on all data across time (since 1999) which means it may be affected by the lower prices in the past 
    - considered training models dynamically, so query data from database depending on user query and then train model on the fly 

6. could potentially use function calling 
