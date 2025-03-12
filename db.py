from pymongo import MongoClient
import pandas as pd

def fetch_for_prediction():
    client = MongoClient("mongodb://localhost:27017")
    db = client["capstone"]

    # Define the fields to project dynamically from debt_info
    debt_info_fields = [
        "Description", "Fixed_Float", "Debt_Currency", "Benchmark_rate", 
        "Maturity_Year", "Maturity_Date", "PSA_Interest", "Proportionate_Consolidation", 
        "Interest_Cost_Fixed", "Interest_Cost_Float", 
        "Scenario", "Currency", "View", "Consolidation"
    ]
    
    # Build the project stage dynamically
    project_stage = {
        "_id": 0,
        "Debt_Info_Key": 1,
        "GFDB_Key": 1,
        "Interest_rate": "$debt_info.Interest_rate",
        "Repayment_Amount": 1,
        "Repayment_Year": 1,
        "Borrowing_Amount": "$debt_info.Borrowing_Amount"
    }
    
    # Add debt_info fields to project dynamically
    project_stage.update({field: f"$debt_info.{field}" for field in debt_info_fields})
    
    # Define the pipeline with the lookup and unwind stages
    pipeline = [
        {
            "$lookup": {
                "from": "debt_information",
                "localField": "Debt_Info_Key",
                "foreignField": "Debt_Info_Key",
                "as": "debt_info"
            }
        },
        {"$unwind": "$debt_info"},
        {"$project": project_stage}
    ]
    
    merged_data = list(db["payment_projection"].aggregate(pipeline))
    return pd.DataFrame(merged_data) if merged_data else None


def fetch_and_merge_data():
    client = MongoClient("mongodb://localhost:27017")
    db = client["capstone"]

    # Fetch payment_projection data
    payment_projection_data = list(db["payment_projection"].find())
    payment_projection_df = pd.DataFrame(payment_projection_data)

    # Fetch debt_information data
    debt_information_data = list(db["debt_information"].find())
    debt_information_df = pd.DataFrame(debt_information_data)

    # Merge the two dataframes based on Debt_Info_Key
    merged_df = pd.merge(payment_projection_df, debt_information_df, on="Debt_Info_Key", how="left")

    return merged_df

