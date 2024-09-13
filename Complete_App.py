
import streamlit as st

# add a logo to the sidebar at the top
st.sidebar.image("Logo_ML_trans_black.png", use_column_width = True)

pg = st.navigation([
    st.Page("Home.py", title = "Home",icon = ":material/home:"),
    st.Page("Online_App.py", title = "Online Application",icon = ":material/manufacturing:"),
    st.Page("Example.py", title = "Example",icon = ":material/question_mark:"),
    st.Page("Contact.py", title = "Contact",icon = ":material/contact_page:"),
])

pg.run()

