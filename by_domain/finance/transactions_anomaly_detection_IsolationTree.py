import kagglehub

# Download latest version
path = kagglehub.dataset_download("agitamuhammad/transactional-anomalies-dataset")

print("Path to dataset files:", path)

if __name__=='__main__':
    print ("Transactions")