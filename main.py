import os
from data_prep import prepare_data
from train_dt import train as train_dt
from train_knn import train as train_knn
from train_svc import train as train_svc
from ensemble import run_ensemble

def main():
    print("==================================================")
    print("STEP 1: Executing Data Preparation...")
    prepare_data()
    
    print("==================================================")
    print("STEP 2: Training Decision Tree Classifier...")
    train_dt()
    
    print("==================================================")
    print("STEP 3: Training K-Nearest Neighbors Classifier...")
    train_knn()
    
    print("==================================================")
    print("STEP 4: Training Support Vector Classifier...")
    train_svc()
    
    print("==================================================")
    print("STEP 5: Executing Ensemble Combination...")
    run_ensemble()
    
    print("==================================================")
    print("Modular Pipeline Orchestration Complete!")

if __name__ == "__main__":
    main()
