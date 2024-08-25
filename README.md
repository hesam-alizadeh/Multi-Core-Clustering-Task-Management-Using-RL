# Task Management and Cloud Computing Using Machine Learning and Multi-Core Clustering .
## Motivation
In today's era of big data, where vast amounts of complex and dynamic data are generated at unprecedented rates, efficiently processing and extracting meaningful insights from this data is crucial. Traditional clustering algorithms, while effective in smaller-scale scenarios, often struggle to cope with the scale, complexity, and velocity of data in modern computing environments, such as cloud and fog computing. These challenges necessitate the development of advanced methods that can handle the demands of big data.

Multi-core clustering algorithms offer a promising solution to this challenge by leveraging parallel processing capabilities to accelerate the clustering process. This is particularly important in task scheduling, where determining the similarity between tasks is critical for optimizing resource allocation, improving performance, and reducing computational overhead. By efficiently clustering tasks that are most similar to each other, multi-core clustering algorithms can significantly enhance the overall efficiency and effectiveness of task scheduling in big data environments.

The motivation behind this approach is to bridge the gap between the growing demands of big data processing and the limitations of traditional clustering techniques. By harnessing the power of multi-core architectures, we aim to develop a clustering framework that not only meets the scalability and speed requirements of big data but also provides accurate and reliable task scheduling solutions. This approach is essential for modern information societies, where the ability to process and analyze large datasets quickly and cost-effectively is a key determinant of success.



## Dataset and Publications Used
- The data is obtained from the following   [`dataset`](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample).

This is a trace of the workloads running on eight Google Borg compute clusters for the month of May 2019. The trace describes every job submission, scheduling decision, and resource usage data for the jobs that ran in those clusters.

It builds on the May 2011 trace of one cluster, which has enabled a wide range of research on advancing the state-of-the-art for cluster schedulers and cloud computing, and has been used to generate hundreds of analyses and studies.

Since 2011, machines and software have evolved, workloads have changed, and the importance of workload variance has become even clearer. The new trace allows researchers to explore these changes. The new dataset includes additional data, including:

CPU usage information histograms for each 5 minute period, not just a point sample;
information about alloc sets (shared resource reservations used by jobs); and
job-parent information for master/worker relationships such as MapReduce jobs.
Just like the last trace, these new ones focus on resource requests and usage, and contain no information about end users, their data, or access patterns to storage systems and other services.

The trace data is being made available via Google BigQuery so that sophisticated analyses can be performed without requiring local resources. This site provides access instructions and a detailed description of what the traces contain.








## Goals of the Project
A logical solution to consider the overlap of clusters is assigning a set of membership degrees to each data point. Fuzzy clustering, due to its reduced partitions and decreased search space, generally incurs lower computational overhead and easily handles ambiguous, noisy, and outlier data. Thus, fuzzy clustering is considered an advanced clustering method. However, fuzzy clustering methods often struggle with non-linear data relationships. This paper proposes a method based on feasible ideas that utilizes multicore learning within the Hadoop map reduce framework to identify inseparable linear clusters in complex big data structures. The multicore learning model is capable of capturing complex relationships among data, while Hadoop enables us to interact with a logical cluster of processing and data storage nodes instead of interacting with individual operating systems and processors. In summary, the paper presents the modeling of non-linear data relationships using multicore learning, determination of appropriate values for fuzzy parameterization and feasibility, and the provision of an algorithm within the Hadoop map reduce model.
The experiments were conducted on one of the commonly used datasets from the ['Google_Research_Machine_Learning_Repository'](https://research.google/tools/datasets/google-cluster-workload-traces-2019/), as well as on the implemented CloudSim dataset simulator, and satisfactory results were obtained.
According to published studies, the Google Research Machine Learning Repository is suitable for regression and clustering purposes in analyzing large-scale datasets, while the CloudSim dataset is specifically designed for simulating cloud computing scenarios, calculating time delays, and task scheduling.

The following steps need to be perfomed :  
1. Develop a Scalable Clustering Framework: Create a multi-core clustering framework that can efficiently handle the scale, complexity, and velocity of big data in modern computing environments, such as cloud and fog computing.

2. Enhance Task Scheduling Efficiency: Improve the efficiency and effectiveness of task scheduling by accurately clustering tasks based on their similarity, optimizing resource allocation, and reducing computational overhead.

3. Integrate Fuzzy Clustering for Ambiguous Data: Incorporate fuzzy clustering techniques to manage ambiguous, noisy, and outlier data, reducing partitions and decreasing search space while maintaining low computational overhead.

4. Model Non-linear Data Relationships: Develop a method to model non-linear data relationships using multi-core learning within the Hadoop MapReduce framework, enabling the identification of inseparable linear clusters in complex big data structures.

5. Utilize Multi-core Learning in Hadoop: Leverage multi-core learning to capture complex relationships among data and use the Hadoop MapReduce framework to interact with logical clusters of processing and data storage nodes efficiently.

6. Optimize Fuzzy Parameterization: Determine appropriate values for fuzzy parameterization and feasibility to enhance the performance and accuracy of the proposed clustering method.

7. Evaluate with Real-world and Simulated Datasets: Conduct experiments using real-world datasets from the Google Research Machine Learning Repository and simulated datasets from CloudSim to validate the effectiveness of the proposed clustering method.

8. Provide a Robust Clustering Algorithm: Develop and present a robust algorithm within the Hadoop MapReduce model that effectively handles large-scale datasets and complex data structures for task scheduling and clustering purposes.



### File Structure  
- `Multi-Core-Clustering.ipynb` -> Jupyter Notebook with ML Model  
- `Google Borg Primary Dataset.parquet` , `Final Dataset.parquet`  -> Dataset
- `Project Report` -> Report specifying the project




### 5.2. ANALYSIS
To evaluate the efficiency of the proposed method, it is compared with three clustering methods: KMeans, MPC-KMeans, and Hierarchical Clustering.

Hierarchical Clustering:
In this category of methods, data is organized in a hierarchical tree structure based on distance criteria. Typically, the approach in hierarchical clustering is based on greedy algorithms. In hierarchical clustering, data is clustered into a hierarchical structure, producing a tree diagram of clusters known as a dendrogram. Cutting the resulting dendrogram at any desired level will result in different clusters. The methods in this category are either divisive or agglomerative. In the divisive method, the dendrogram clusters are built top-down. In these methods, all the data is first placed in a single cluster, and then the cluster is divided according to suitable criteria until the convergence condition is met, along with constructing the cluster dendrogram. Unlike divisive methods, in agglomerative methods, the dendrogram clusters are built bottom-up. These methods consider each data point as a cluster at the lowest level and continue merging clusters until the stopping condition is met.

The BIRCH clustering algorithm is one of the prominent methods available in this category of clustering methods.

The experimental results are reported based on ARI queries and standard deviation. The standard deviation of ARI is one of the dispersion indicators that shows how much, on average, the data deviates from the mean. If the standard deviation of a data set is close to zero, it indicates that the data points of different clusters are close to each other and have little dispersion, while a large standard deviation indicates significant data dispersion. The kernel matrices {K1_v, K2_v, …, K5_v, K1_g, K2_g, …, K7_g} have been considered as the base kernels in all experiments.

![image](https://github.com/user-attachments/assets/31a2bb5e-230e-4d89-85bc-4c82a7131e7e)



The analysis and comparison of the proposed method with other methods were conducted based on metrics such as the standard deviation and the number of queries. As observed in the behavior of other methods shown in the charts in Figure 4, it can be inferred that the proposed method is more efficient than other clustering methods like KMeans, MPC-KMeans, and hierarchical clustering. This indicates the high reliability of the proposed method. The superiority of the proposed method compared to the KMeans method can be attributed to its nonlinear nature, which, by utilizing a multi-core learning model within the framework of mapping reduction, is capable of discovering complex relationships between data, while the KMeans method suffers from the lack of this feature.

When comparing the proposed method with the hierarchical method, it should be noted that the proposed method has demonstrated much higher efficiency compared to the hierarchical method. Therefore, it can be concluded that the hierarchical method is not efficient enough for clustering data with complex structures or high dimensions. The inability to model nonlinear relationships in the data by the hierarchical method can be considered one of the reasons for its inefficiency.

In comparing the proposed method with the MPC-KMeans method, which overall has provided better results than the KMeans and hierarchical methods, it can be said that this superiority is more evident in higher-dimensional datasets. Although MPC-KMeans attempts to learn the correct metric during clustering, experiments have shown that the efficiency of this method does not significantly increase when the number of queries exceeds a certain threshold. As shown in Figure 5, as the number of queries exceeds a certain threshold, the efficiency of the MPC-KMeans method remains at a relatively constant and non-increasing level. In contrast, in the proposed method, we observe an increasing trend in efficiency with the increase in the number of queries, which indicates the effectiveness of the selected features and the high learning capability of the proposed method.

Determining the appropriate values for the weighting coefficients m and p is an open issue in the field of fuzzy clustering. For example, Table 1 shows the efficiency of the proposed method considering values of m=p=3 and m=p=1.3. According to this figure, fuzzier clustering has provided better results.

![image](https://github.com/user-attachments/assets/f0c1784b-3054-40ea-b476-3676ff82e190)

Due to the alignment of the nature of data related to task scheduling concepts with the features of the data focused on in this article, the proposed method was also implemented on the CloudSim simulated dataset. CloudSim is a platform that has recently attracted the attention of researchers for simulating cloud computing scenarios, calculating time delays, and task scheduling. The required data was obtained through contact with one of the authors of reference [32]. This data included various features and information, some of the key features of which are:

- Network topology: Information related to the network topology, showing how servers and clouds are interconnected and how traffic flows within the network.
- Servers and virtual machines: Specifications of the servers and virtual machines that represent computing resources in the cloud environment, including the number of processing cores, memory capacity, disk space, network bandwidth, and other hardware-related features.
- Users and requests: Information about users and the requests they make to utilize cloud services, including request type, scheduling, the amount of requested resources, and other user-related features.
- Traffic patterns: Various traffic patterns generated by users while utilizing cloud services, including resource usage patterns, traffic variation patterns, and different loading patterns.
- Scheduling and events: Information related to the scheduling of activities and various events in the cloud environment, including request scheduling, traffic variation scheduling, the scheduling of creating and removing servers and virtual machines, and other related events.

After conducting the experiments, it was observed that the results obtained were highly consistent and similar to the results achieved with the previous dataset. These results indicate the efficiency and applicability of the proposed method in this article.






## 6. CONCLUSION

### 6.1. LEARNING FROM THE PROJECT
This article presents a method based on multi-core fuzzy clustering for clustering big data using the MapReduce model of Hadoop. The main focus of this article on fuzzy clustering is due to the overlapping of clusters and the generalization power of fuzzy logic in dealing with noisy and outlier data. The work done in this area is introduced, and their superior points are highlighted. The proposed method is introduced within the framework of Hadoop's MapReduce model. Hadoop enabled us to interact not with the operating system and processor but with a logical cluster of processes and data warehouse nodes. The two important components of Hadoop are HDFS, which supports petabytes of data, and scalable MapReduce, which computes results in batches. To detect linearly inseparable clusters in complex big data structures, the proposed objective function is considered in a multi-core learning architecture and implemented within the Hadoop MapReduce framework. The proposed method is secured against inappropriate functions or irrelevant features by automatically adjusting the weight of the kernels and considering a penalty term. This reduced the sensitivity of the proposed method to the selection of inefficient kernels. The performance of the proposed method was compared with three clustering methods: KMeans, MPC-KMeans, and hierarchical clustering. The results from the experiments show that our dataset had a high level of overlap and similarity, which indicates the efficiency and applicability of the proposed method in this article.

Determining the appropriate value for the parameter controlling the degree of fuzziness in clustering (m) and the parameter for the weight of membership feasibility (p) remains an open problem in the field of fuzzy clustering. Therefore, this article also conducted experiments to determine the appropriate values for these parameters. The results indicate that increasing the degree of fuzziness in clustering provides better results. The high performance of the proposed method compared to other clustering methods (KMeans, MPC-KMeans, and hierarchical clustering) was very evident in the experiments conducted, as analyzed in the previous section. Among the most important reasons for the proposed method in clustering big data are the ability to model nonlinear relationships in the data using a multi-core learning model, determining suitable values for fuzzification and feasibility parameters, and presenting the algorithm within the MapReduce model.



### 6.2 WORK TO BE DONE IN THE FUTURE
Here are the following suggestions for future work:

1. Since the proposed approach is based on linear metric learning, one of the limitations of the model could be situations where the problem space is non-linear. In such cases, the problem can be solved using non-linear kernels or multiple linear metric methods.
2. The presence of noisy or outlier data can affect the performance of the proposed model. This issue can be partially addressed by employing various methods for detecting such data, using models that are more robust to these types of data, or by adjusting some of the model's hyperparameters.
3. Since determining the optimal center for clusters is a critical issue, using some metaheuristic methods such as genetic algorithms or particle swarm optimization can significantly help in finding the optimal or faster cluster centers.
4. Using other similarity metrics instead of Euclidean distance, such as Mahalanobis or Manhattan distance, might lead to better results in some cases for the model's performance.
5. As we have seen before, selecting the appropriate type and parameters for the kernel plays an important role in the efficiency of this model. Therefore, conducting a thorough search in the space of these values and finding the optimal values is one of the issues that can be considered in future work.



## 7. REFERENCES
1. S.M. Razavi, M. Kashani, S. Paydar, “Big Data Fuzzy C-Means Algorithm based on Bee Colony Optimization using an Apache Hbase”, Journal of Big Data, Vol. 8, Article Number: 64, 2021
2. S. Sinha, “Hadoop Ecosystem: Hadoop Tools for Crunching Big Data”, edureka, https://www.edureka.co/blog/hadoop-ecosystem, 2022.
3. S. Landest, T. khoshgoftaar, A.N. Richter, “A Survey of Open Source Tools for Machine Learning with Big Data in the Hadoop Ecosystem”, Journal of Big Data, Vol. 2, No.1, 2015.
4. X. Liu, X. Zhu, M. Li, L. Wang, E. zhu, T. Liu, M. Kloft, D. Shen, J. Yin, W. Gao, “Multiple Kernel k-Means with Incomplete Kernels”, IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 42, No. 5, pp.1191-1204, 2020.
5. R. K. Sanodiya, S. Saha, J. Mathew, “A Kernel Semi-Supervised Distance Metric Learning with Relative Distance: Integration with a MOO Approach”, Expert Systems with Applications, Elsevier, Vol. 125, pp. 233-248, 2019.
6. M. Soleymani Baghshah, S. Bagheri Shouraki, “Efficient Kernel Learning from Constraints and Unlabeled Data”, 20th International Conference on Pattern Recognition, Istanbul, Turkey, pp. 3364-3367, 2010.
7. S. Zhu, D. Wang, T. Li, “Data Clustering with Size Constraints”, Knowledge-Based Systems, Elsevier, Vol. 23, pp. 883-889, 2010.
8. L. A. Maraziotis, “A Semi-Supervised Fuzzy Clustering Algorithm Applied to Gene Expression Data”, Pattern Recognition, Elsevier, Vol. 45, pp. 637-648, 2014.
9. J. Bezdek, R. Ehrlich, W. Full, “FCM: the Fuzzy C-Means Clustering Algorithm”, Computers & Geosciences, Elsevier Vol. 10, Issue. 2-3, pp. 191-203, 1984.
10. O. Ozdemir, A. Kaya, “Comparison of FCM, PCM, FPCM and PFCM Algorithms in Clustering Methods”, Afyon Kocatepe University Journal of Science and Engineering, pp. 92-102, 2019.
11. M. A. Lopez Felip, T. J. Davis, T. D. Frank, J.A. Dixon, “A Cluster Phase Analysis for Collective Behavior in Team Sports”, Human Movement Science, Elsevier, Vol. 59, pp. 96-111, 2018.
12. F. Hai Jun, W. Xiao Hong, M. Han Ping, W. Bin, “Fuzzy Entropy Clustering using Possibilistic Approach”, Advanced in Control Engineering and Information Science, Elsevier, Procedia Engineering Vol. 15, pp.1993-1997, 2011.
13. M. Bouzbida, L. Hassine, A. Chaari, “Robust Kernel Clustering Algorithm for Nonlinear System Identification”, Hindawi, Mathematical Problems in Engineering, pp. 1-11, 2017.
14. T.H. Sardar, Z. Ansari, “MapReduce-based Fuzzy C-means Algorithm for Distributed Document Clustering”, Journal of The Institution of Engineers (India): Series B, Vol. 103, No. 1, pp.131-142, 2022.
15. Q. Yu, Z. Ding, “An Improved Fuzzy C-Means Algorithm based on MapReduce”, 8th International Conference on Biomedical Engineering and Informatics (BMEI), pp. 634-638, 2015.
16. J. Dean, S. Ghemawat, “MapReduce: Simplified Data Processing on Large Clusters”, Sixth Symposium on Operating System Design and Implementation, San Francisco, CA, pp. 137-150, 2004.
17. L. Jiamin and F. Jun, “A Survey of MapReduce based Parallel Processing Technologies”, China Communications, Vol. 11, No. 14, pp. 146–155, 2014.
18. W. Zhao, H. Ma, Q. He, “Parallel K-Means Clustering based on MapReduce, in Cloud Computing”, IEEE International Conference on Cloud Computing, pp. 674-679, Part of the Lecture Notes in Computer Science book series (LNCS, volume 5931), 2009.
19. H. Bei, Y. Mao, W. Wang, X. Zhang, “Fuzzy Clustering Method Based on Improved Weighted Distance”, Mathematical Problem in Engineering, Vol. 5, Hindawi, 2021.
20. S.A.Ludwig, “MapReduce-based Fuzzy C-Means Clustering Algorithm: Implementation and Scalability”, International Journal of Machine Learning and Cybernetics, pp. 923-934, Copyright owner: Springer-Verlag Berlin Heidelberg, 2015.
21. J. Ramisingh, V. Bhuvaneswari, “An Integrated Multi-Node Hadoop Framework to Predict High-Risk Factors of Diabetes Mellitus using a Multilevel MapReduce based Fuzzy Classifier (MMR-FC) and Modified DBSCAN Algorithm”, Applied Soft Computing, Vol. 108, 2021.
22. T.H. Sardar, Z. Ansari, “Partition based clustering of large datasets using MapReduce framework: An analysis of recent themes and directions”, Future Computing and Informatics Journal, Vol. 3, No. 2, pp. 247-261, 2018.
23. A. A. Abin, H. Beigy, “Active Constrained Fuzzy Clustering: A Multiple Kernels Learning Approach”, Pattern Recognition, Elsevier, Vol. 48, Issue. 3, pp. 935-967, 2015.
24. H. Hassani, M. Kalantari, C. Beneki, “Comparative Assessment of Hierarchical Clustering Methods for Grouping in Singular Spectrum Analysis”, AppliedMath, Vol. 1, No.1, pp. 18-36, 2021.
25. S.A. Elavarasi, D.J. Akilandeswari, D.B. Sathiyabhama, “A Survey on Partition Clustering Algorithms”, International Journal of Enterprise Computing and Business Systems, Vol.1, pp.1–14, 2011.
26. T. Zhang, R. Ramakrishnan, M. Livny, “Birch: an Efficient Data Clustering Method for Very Large Databases”, SIGMOD Record, Vol.25, No.2, pp.103–114, 1996.


27. C. Swafford, “Red Wine Quality Analysis”, https://rpubs.com/cswaff7/775970, 2021.
28. M. A. Ala’anzy, M. Othman, Z. M. Hanapi, M. A. Alrshah, “Locust Inspired Algorithm for Cloudlet Scheduling in Cloud Computing Environments”, Sensors, Vol. 21, No. 21, 19 Pages, 2021.
29. R. Mahmud, S. Pallewatta, M. Goudarzi, R. Buyya, “iFogSim2: An Extended iFogSim Simulator for Mobility, Clustering, and Microservice Management in Edge and Fog Computing Environments”, The University of Melbourne, Journal of Systems and Software, Vol. 190, 2022.
30. H. Gupta, A.V. Dastjerdi, S.K. Ghosh, R. Buyya, “iFogSim: A toolkit for Modeling and Simulation of Resource Management Techniques in the Internet of Things, Edge and Fog Computing Environments”, Cloud and Fog Computing, Volume 47, Issue 9, Pages 1275-1296, 2017.
