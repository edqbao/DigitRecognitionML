filename = "/Users/kat/Desktop/Kaggle/Graph.csv";
graph_data = csvread(filename);
adj_mtx = zeros(6000,6000);
for i = 1:1:length(graph_data)
     adj_mtx(graph_data(i,1), graph_data(i,2)) = 1;
     adj_mtx(graph_data(i,2), graph_data(i,1)) = 1;
end

Sigma = cov(features);
Mu = mean(features);
K = 1035;
[W_1035,E] = eigs(Sigma, K);
Y_1035 = (features-repmat(Mu,[10000,1]))*W_1035;
Xhat_1035 = Y_1035*W_1035' + repmat(Mu,[10000,1]);
csvwrite('reduced_3000.csv',Xhat);
