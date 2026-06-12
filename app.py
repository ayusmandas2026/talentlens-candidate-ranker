import streamlit as st

# recommendation badges mapping:
# Strong Hire 1-5, Hire 6-20, Borderline 21-50
def get_recommendation_badge(rank):
    if rank <= 5: return "Strong Hire"
    elif rank <= 20: return "Hire"
    elif rank <= 50: return "Borderline"
    else: return "No Hire"
