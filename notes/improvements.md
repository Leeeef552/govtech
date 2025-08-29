1. use OneMAP api to draw distance between point and the nearest amenities to the location as features for price prediction
    - use more features like remaining lease which are all predictive features in predicting price
    - tried to implement nearest mrt and public transport but api rate limited and it takes too long

2. under the resale there is a column remaining lease, i think can potentially add to the other columns to match but will leave out for now

5. another assumption is the recency of the data, one model is trained on all data across time (since 1999) which means it may be affected by the lower prices in the past 
    - considered training models dynamically, so query data from database depending on user query and then train model on the fly 

