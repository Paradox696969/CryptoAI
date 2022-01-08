In the future, this retraining script will be used as the current amount of data is insufficient,
 and the main site's API is not developed enough to return collection data at the moment. 
For now, this script will stay here until we have time to complete this section of the project.

If you are curious how this section will work, then here you go. This script will request the main site to return collection data every 12 hours.
It will then train the models required. When a prediction requiring these models is needed, then the main site requests for the prediction to be made, 
then the prediction is sent to the main site. When requesting for a prediction via API requests, it may be more beneficial to request directly from the webserver
the retraining script is hosted on.