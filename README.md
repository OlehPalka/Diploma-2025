Diploma Project

Data
Due to the company's policy on sensitive user data (name, surname, phone number, address, etc.), datasets and Tableau dashboards are not provided here. If there is a need to get initial data from Matomo, pre-processed data tables from Tableau, or results of algorithms execution, You can reach me, Oleh Palka oleh.palka@ucu.edu.ua or Lidia Bedrijchuck at lida.bedriychuk@kormotech.com.ua (data analyst from the Kormotech team).

Code explanation
db_creation.py - file to set up local MySQL DB to extract initial data from Matomo
create_product_table.py - file to create product table out of purchases data
user_timings.py - file to extract day and hour when user visited page most of the times
correlations.py - file that runs correlation analysis of User table of metrics which would be used in clustering
PCA.py - PCA analysis to select only valuable metrics for clustering
klustering_parameters_selection - file that runs and visualises clustered data based on different parameters. This is done to select the best parameters that perform great clustering.
klustering - file with identified best parameters for each clustering method. 
k-means - file with detailed analysis of K-means, an algorithm that performed the best.
users_clusters_assigning.py - code that assigns clusters to the New users group and the Churned users group
offers_files_creation.py - creation of personalized offers for each user from selected groups.
ab_test_groups_creation.py - creating of two groups (test and control) for both New users and Churned ones.
create_AB_test_groups_for_march.py - selecting users from March to perform test on them.
gathering_ab_test_results.py - gathering ab test results from April data.
ab_test_metrics_difference.py - printing out difference between test and control groups based on selected metrics.
purchased_users_in_april.py - selection of users who bought after messaging to check whether they subscribed.

