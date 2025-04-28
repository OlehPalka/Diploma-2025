Diploma Project

Data
Due to the company's policy on sensitive user data (name, surname, phone number, address, etc.), datasets and Tableau dashboards are not provided here. If there is a need to get initial data from Matomo, pre-processed data tables from Tableau, or results of algorithms execution, You can reach me, Oleh Palka oleh.palka@ucu.edu.ua or Lidia Bedrijchuck at lida.bedriychuk@kormotech.com.ua (data analyst from the Kormotech team).

**Code explanation** 
____________________
**db_creation.py**  
File to set up local MySQL DB to extract initial data from Matomo.

**create_product_table.py**  
File to create product table out of purchases data.

**user_timings.py**  
File to extract day and hour when user visited the page most often.

**correlations.py**  
File that runs correlation analysis of the User table metrics which will be used in clustering.

**PCA.py**  
PCA analysis to select only the most valuable metrics for clustering.

**klustering_parameters_selection.py**  
File that runs and visualizes clustered data based on different parameters to identify the best-performing configuration.

**klustering.py**  
File with the identified best parameters for each clustering method.

**k-means.py**  
File with detailed analysis of K-means, the algorithm that performed best.

**users_clusters_assigning.py**  
Code that assigns clusters to the New Users group and the Churned Users group.

**offers_files_creation.py**  
Creates personalized offers for each user in the selected groups.

**ab_test_groups_creation.py**  
Creates two groups (test and control) for both New Users and Churned Users.

**create_AB_test_groups_for_march.py**  
Selects users from March to include in the A/B test.

**gathering_ab_test_results.py**  
Gathers A/B test results from April data.

**ab_test_metrics_difference.py**  
Prints the difference between test and control groups based on selected metrics.

**purchased_users_in_april.py**  
Selects users who made a purchase after messaging to check whether they subscribed.
