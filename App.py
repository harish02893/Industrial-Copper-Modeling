import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
#st.set_page_config(layout="wide")
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
import os
import pickle




#application background
def app_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://tse2.mm.bing.net/th?id=OIP.-Omurt7RrwN6iJ2FfrSp3gHaFj&pid=Api&P=0&h=220");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True)
app_bg()



st.write("""
<div style='text-align:center; background-color:#009999;'>
    <h1 style='color:#FFFFFF; font-family: Rockwell, sans-serif; font-weight: bold;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)



# Define custom CSS styles for the tabs
tab_style = """
    font-size: 30px;
    font-family: Rockwell, Rockwell;
    color: black;
    text-align: center;
"""


# Create tabs
tab1, tab2 = st.columns(2)
# Render the tabs with custom styling
with tab1:
    st.markdown("<div style='" + tab_style + "'>PREDICT SELLING PRICE</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div style='" + tab_style + "'>PREDICT STATUS</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"]) 
with tab1:    
        

        # Define the possible values for the dropdown menus
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
            with col3:               
                st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
             
        if submit_button and flag==0:
            
            
           
            
            import pickle
            model_file_path = "D:\\copper project\\model.pkl"
            with open(model_file_path, 'rb') as file:
                # Your code to handle the file

            #with open(r"model/model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
 
            model_file_path = "D:\\copper project\\scaler.pkl"
            with open(model_file_path, 'rb') as file:
            #with open(r'model/scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(file)

            
            model_file_path = "D:\\copper project\\t.pkl"
            with open(model_file_path, 'rb') as file:
            #with open(r'model/scaler.pkl', 'rb') as f:
                t_loaded = pickle.load(file)        

            model_file_path = "D:\\copper project\\/s.pkl"
            with open(model_file_path, 'rb') as file:
            #with open(r'model/scaler.pkl', 'rb') as f:
                s_loaded = pickle.load(file) 
           
                



            


            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
            
with tab2: 
    
        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
             
        if csubmit_button and cflag==0:
            import pickle
            
            
            model_file_path = "D:\\copper project\\cmodel.pkl"
            with open(model_file_path, 'rb') as file:
            #with open(r'model/scaler.pkl', 'rb') as f:
                cloaded_model = pickle.load(file)      
           
            model_file_path = "D:\\copper project\\cscaler.pkl"
            with open(model_file_path, 'rb') as file:
            #with open(r'model/scaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(file)              
            
            model_file_path = "D:\\copper project\\ct.pkl"
            with open(model_file_path, 'rb') as file:
            #with open(r'model/scaler.pkl', 'rb') as f:
                ct_loaded = pickle.load(file)             
            
            
            
            
            
            



            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            #st.write(new_pred)
            if new_pred.all()==1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
                



import datetime

current_date_time = datetime.datetime.now()

st.write(f'<h6 style="color:black;">Kesavan sekar DT1819 | {current_date_time}</h6>', unsafe_allow_html=True)



import streamlit as st

def main():
    #st.title("Kesavan sekar DT1819")

    # Define image URLs
    images = [
        "https://tse2.mm.bing.net/th?id=OIP.hAojpTfiSG_elgZ8QPdKeAHaDt&pid=Api&P=0&h=220",
        "https://tse4.explicit.bing.net/th?id=OIP.akghOiuTLD5tuUc-sE0n6QHaE7&pid=Api&P=0&h=220",
        "https://www.rkmi.co.in/images/commerical-copper-coil2.jpg",
        "https://4.imimg.com/data4/EH/WH/MY-2772533/industrial-copper-tubes-500x500.jpg",
        "https://public.blenderkit.com/thumbnails/assets/2d412869cca14739a7a42659cb9b2058/files/thumbnail_dd3d74f6-732f-4d5a-b6d8-0848c345a207.png.512x512_q85.png",
        "https://5.imimg.com/data5/SELLER/Default/2021/10/IP/RQ/TF/75031140/industrial-copper-tube-1000x1000.jpg",

    ]

    # Create HTML for scrolling images
    html = """
    <style>
    #footer {
        width: 100%;
        overflow: hidden;
        position: fixed;
        bottom: 0;
        background-color: #f1f1f1;
        white-space: nowrap;
        height: 150px; 
    }

    .scrolling-wrapper {
        animation: scroll 30s linear infinite;
    }

    @keyframes scroll {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    .image-wrapper {
        display: inline-block;
        margin: 0 20px;
    }

    .image-wrapper img {
        width: 150px; /* Set width of image */
        height: auto; /* Maintain aspect ratio */
    }
    </style>

    <div id="footer">
        <div class="scrolling-wrapper">
            """
    for image in images:
        html += f'<div class="image-wrapper"><img src="{image}" alt="image"></div>'
    html += """
        </div>
    </div>
    """

    # Display scrolling images using st.markdown
    st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

