import streamlit as st

# create title
st.title("Contact")

# create info text

col1, col2 = st.columns(2)

with col2:
    st.image("Logo_ML_trans_black.png", use_column_width = True)
    
with col1:
    
    st.markdown(
        """
        <br><br>
        If you experience any problems with our application, have questions, or need more
        information regarding copyright regulations, please don't hesitate to contact us!
        
        *TRUMA Technology*  
        Grindelberg 22  
        20144 Hamburg  
        Tel: 0402764182  
        E-Mail: truma@technology.com
        """, 
        unsafe_allow_html=True
    )