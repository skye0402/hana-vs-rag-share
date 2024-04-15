from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from gradio import Blocks
import os, logging
from operator import itemgetter
import requests
from requests.auth import HTTPBasicAuth


from gen_ai_hub.proxy.core import set_proxy_version
from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.proxy.langchain import init_llm
from langchain.globals import set_debug
# set_debug(True)

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List

from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        )
from langchain.memory import ConversationBufferMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableSerializable
from langchain.schema.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB

from hana_ml import ConnectionContext
from hdbcli.dbapi import Connection

block_css = """
gradio-app > .gradio-container {
    max-width: 100% !important;
    
}
.contain { display: flex !important; flex-direction: column !important; }
#chat_window { height: 82vh !important; }
#column_left { height: 88vh !important; }
#sql_col_left1 { height: 82vh !important;}
#arch_gallery { height: 88vh !important;}
#buttons button {
    min-width: min(120px,100%);
}
footer {
    display:none !important
}
"""

LOGO_MARKDOWN = f"""
### APJ Architecture & Platform Advisory
![Platform & Integration Labs](file/img/blue.svg)
"""

global GENAIHUB_PROXY_CLIENT
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","all-MiniLM-L12-v2") 
DEFAULT_EF = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
S4PROD_URL = "https://my300181.s4hana.ondemand.com/ui#Material-displayFactSheet?Product="
TABLE_NAME = "S4PRODUCTDATA"

SYS_TEMPLATE = """
    You are a helpful assistant to identify the best product matching to a search. Don't make up anything. If there is no match say that there is no match.
    Pass back the product name as a link in markdown language. For example if the product id is 'APJPIL202403111229' the url to the link is 'https://my300181.s4hana.ondemand.com/ui#Material-displayFactSheet?Product=APJPIL202403111229'. 
    Also add the description and why you think it's a good match to the user's question. Don't output products that don't match.
"""

HUMAN_TEMPLATE = """
    The question is: {query}. Return a list of bullet points for each matching product. Use Markdown to create a link to the product. If the product is no good match, don't output it at all and don't mention it.
    ===========================
    {context}        
"""

MODELS = [
    {
        "model": "gpt-35-turbo-16k", 
        "name": "Azure OpenAI GPT-3.5 Turbo 16k", 
        "desc": "GPT-3.5 models can understand and generate natural language or code. The most capable and cost effective model in the GPT-3.5 family is gpt-3.5-turbo which has been optimized for chat using the Chat Completions API but works well for traditional completions tasks as well.", 
        "platform": "SAP Generative AI Hub"
    },
    {
        "model": "gpt-4-32k", 
        "name": "Azure OpenAI GPT-4 32k", 
        "desc": "GPT-4 is a large multimodal model (accepting text or image inputs and outputting text) that can solve difficult problems with greater accuracy than any of our previous models, thanks to its broader general knowledge and advanced reasoning capabilities.", 
        "platform": "SAP Generative AI Hub"
    }
    ]

PRODUCT_BUTTON_EN = "Load S/4HANA Products"
PRODUCT_BUTTON_DIS = "Loading products..."

class MatchingProducts(BaseModel):
    products: List[TypedDict("products", {"product_id": str, "comment": str})] = Field(description="The best matching product ids and a comment why they match.") # type: ignore

def get_model(model_info: dict, temperature: float = None, top_p: float = None)->any:
    """ Serve the required LLM chat model """
    active_proxy = None
    model_name = None
    deployment_id = None
    if model_info["platform"] == "SAP Generative AI Hub":
        active_proxy = GENAIHUB_PROXY_CLIENT
        set_proxy_version("gen-ai-hub")
        model_name = model_info["model"]
    else:
        raise Exception("This platform doesn't exist.")
    init_llm_kwargs = {
        "model_name": model_name,
        "deployment_id": deployment_id,
        "proxy_client": active_proxy,
        "max_tokens": 3000
    }
    if temperature is not None:
        init_llm_kwargs["temperature"] = temperature
    if top_p is not None:
        init_llm_kwargs["top_p"] = top_p
    return init_llm(**init_llm_kwargs)

def retrieve_data(vector_db: HanaDB, llm: BaseLanguageModel)->RunnableSerializable:
    """ Retrieves data from store and passes back result """
    retriever = vector_db.as_retriever(k=4)
    my_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(SYS_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)  
        ],
        input_variables=['query', 'context', 'language'],
    )
    rag_chain = (
            {
                "query": itemgetter("query"), 
                "language": itemgetter("language"),
                "context": itemgetter("query") | retriever
            } 
            | my_prompt 
            | llm
        )
    return rag_chain

def get_hana_connection(conn_params: dict)->Connection:
    """ Connect to HANA Cloud """
    connection = None  
    try:
        conn_context = ConnectionContext(
            address = conn_params["host"],
            port = 443,
            user = conn_params["user"],
            password= conn_params["password"],
            encrypt= True
        )    
        connection = conn_context.connection
        logging.info(conn_context.hana_version())
        logging.info(conn_context.get_current_schema())
    except Exception as e:
        logging.error(f'Error when opening connection to HANA Cloud DB with host: {conn_params["host"]}, user {conn_params["user"]}. Error was:\n{e}')
    finally:    
        return connection

def get_product_data(base_url, product_filter, user, password, max_products=None)->list:
    """ Returns a selection of the product master from S/4HANA """
    # Set up the session with basic authentication
    session = requests.Session()
    session.auth = HTTPBasicAuth(user, password)
    # Define the headers to request JSON response
    headers = {
        'Accept': 'application/json'
    }
    # Make the initial request to get the list of products
    product_list_url = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product?$filter=startswith(Product,'{product_filter}')"
    product_list_response = session.get(product_list_url, headers=headers)
    product_list_response.raise_for_status()  # Raise an error for bad status codes
    product_list = product_list_response.json()['d']['results']
    # Limit the number of products to process if max_products is set
    if max_products:
        product_list = product_list[:max_products]
    # Iterate over the products and fetch their descriptions and sales text
    for product in product_list:
        product_id = product['Product']
        description_url = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product('{product_id}')/to_Description"
        sales_text_url = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_ProductSalesDelivery(Product='{product_id}',ProductSalesOrg='1710',ProductDistributionChnl='10')/to_SalesText"
        uom_url = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product('{product_id}')/to_ProductUnitsOfMeasure"        
        # Fetch the product description in English
        description_response = session.get(description_url, headers=headers)
        description_response.raise_for_status()
        descriptions = description_response.json()['d']['results']
        product['ProductDescription'] = next((desc['ProductDescription'] for desc in descriptions if desc['Language'] == 'EN'), None)
        # Fetch the sales text in English
        sales_text_response = session.get(sales_text_url, headers=headers)
        sales_text_response.raise_for_status()
        sales_texts = sales_text_response.json()['d']['results']
        product['SalesText'] = next((text['LongText'] for text in sales_texts if text['Language'] == 'EN'), None)        
        # Fetch the unit of measure data
        uom_response = session.get(uom_url, headers=headers)
        uom_response.raise_for_status()
        uom_data = uom_response.json()['d']['results']
        if uom_data:
            uom_info = uom_data[0]  # Use the first entry in the list
            product['UnitOfMeasure'] = uom_info['AlternativeUnit']
            product['MaterialVolume'] = uom_info['MaterialVolume']
            product['VolumeUnit'] = uom_info['VolumeUnit']
            product['GrossWeight'] = uom_info['GrossWeight']
            product['WeightUnit'] = uom_info['WeightUnit']
            product['GlobalTradeItemNumber'] = uom_info['GlobalTradeItemNumber']
            product['UnitSpecificProductLength'] = uom_info['UnitSpecificProductLength']
            product['UnitSpecificProductWidth'] = uom_info['UnitSpecificProductWidth']
            product['UnitSpecificProductHeight'] = uom_info['UnitSpecificProductHeight']
            product['ProductMeasurementUnit'] = uom_info['ProductMeasurementUnit']
    
    product_in_text = []
    for product in product_list:
        product_id = product['Product']
        weight = product.get('GrossWeight', 'N/A')
        weight_unit = product.get('WeightUnit', '')
        volume = product.get('MaterialVolume', 'N/A')
        volume_unit = product.get('VolumeUnit', '')
        gtin = product.get('GlobalTradeItemNumber', 'N/A')
        sales_text = product.get('SalesText', 'N/A')
        product_description = product.get('ProductDescription', 'N/A')
        length = product.get('UnitSpecificProductLength', 'N/A')
        width = product.get('UnitSpecificProductWidth', 'N/A')
        height = product.get('UnitSpecificProductHeight', 'N/A')
        dim_uom = product.get('ProductMeasurementUnit', 'N/A')
        # Construct the descriptive text
        descriptive_text = (
            f"Information about Product '{product_id}': "
            f"Weight {weight} {weight_unit}, "
            f"Volume {volume} {volume_unit}, "
            f"Product dimensions: Length {length} {dim_uom} Width {width} {dim_uom} Height {height} {dim_uom}, "
            f"GlobalTradeItemNumber is '{gtin}', "
            f"Sales text: '{sales_text}', "
            f"Product description: '{product_description}'."
        )        
        # Append the product information in text format to the list
        product_in_text.append(
            Document(page_content=descriptive_text, metadata={"product_id": product_id})
        )    
    return product_in_text

def embed_product_data(conn: Connection, embed: SentenceTransformerEmbeddings, s4data: list):
    success = True
    vector_db = HanaDB(
        embedding=embed, connection=conn, table_name=TABLE_NAME
    )
    # Remove existing embeddings from Hana Cloud table (TODO: this needs a nicer logic in a bigger PoC)
    try:
        vector_db.delete(filter={})
    except Exception as e:
        logging.info(f"Deleting embedding entries failed with error {e}. Maybe there were no embeddings?")
    # Add the product data from S/4HANA
    try:
        vector_db.add_documents(s4data)
    except Exception as e:
        logging.error(f"Adding product embeddings failed with error {e}.")
        success = False
    finally:
        return success

def clear_data(state):
    """ Clears the history of the chat """
    state_new = {
        "models": state.get("models"), "connection": state.get("connection", None), 
    }
    return [None, state_new]

def user(state, user_message, history):
    """ Handle user interaction in the chat window """
    state["skip_llm"] = False
    if len(user_message) <= 0:
        state["skip_llm"] = True
        return "", history, None
    rv =  "", history + [[user_message, None]]
    return rv

def call_llm(state, history, model_name):
    """ Handle LLM request and response """
    if state["skip_llm"] == True:
        return history
    history[-1][1] = ""
    model_info = next((item for item in state["models"] if item["name"] == model_name), None)
    llm = state.get("model")
    if llm == None or state.get("memory", None) == None:
        state["model"] = get_model(
            model_info=model_info, 
            temperature=0.0, 
            top_p=0.6
        )
    state["memory"].clear() # In this scenario we don't need a history
    if not state.get("connection"):
        state["connection"] = get_hana_connection(conn_params=state["conn_data"])        
    # Query the product master
    vector_db = HanaDB(connection=state["connection"], embedding=DEFAULT_EF, table_name="S4PRODUCTDATA")
    query = history[-1][0]
    rag_chain = retrieve_data(vector_db=vector_db, llm=state["model"])
    try:
        for response in rag_chain.stream({"query": query, "language": "en"}):
            history[-1][1] += response.content
            logging.debug(history[-1][1])
            yield history
    except Exception as e:
        history[-1][1] += str(f"😱 Oh no! It seems the model {model_info['name']} has some issues. Error was: {e}.")
    state["memory"].save_context({"input": history[-1][0]},{"output": history[-1][1]})
    return history 

def disable_button()->gr.Button:
    return gr.Button(value=PRODUCT_BUTTON_DIS, interactive=False)
        
def build_chat_view(models: dict, conn_data: dict)->Blocks:
    """ Build the view with Gradio blocks """
    with gr.Blocks(
            title="Architecture & Platform Advisory - S/4HANA Product Data search with HANA Vector Store Engine", 
            theme=gr.themes.Soft(),
            css=block_css
        ) as chat_view:
        state = gr.State({})
        state.value["conn_data"] = conn_data
        with gr.Row(elem_id="overall_row") as main_screen:
            with gr.Column(scale=10, elem_id="column_left"):
                with gr.Tab(label="Generative AI", elem_id="genai_tab", visible=True) as genai_tab:
                    chatbot = gr.Chatbot(
                        label="S/4HANA Product Chat",
                        elem_id="chat_window",
                        bubble_full_width=False,
                        show_copy_button=True,
                        show_share_button=True,
                        avatar_images=(None, "./img/saplogo.png")
                    )
                    with gr.Row(elem_id="audio_row1") as query_row:
                        msg_box = gr.Textbox(
                            scale=9,
                            elem_id="msg_box",
                            show_label=False,
                            max_lines=5,
                            placeholder="Enter text and press ENTER",
                            container=False,
                            autofocus=True
                        )
            with gr.Column(scale=3, elem_id="column_right") as column_right:
                with gr.Group(elem_id="model_group") as model_group:
                    model_names = [item["name"] for item in models]
                    state.value["models"] = models
                    model_selector = gr.Dropdown(
                        choices=model_names, 
                        container=True,
                        label="🗨️ Language Model",
                        show_label=True,
                        interactive=True,
                        value=model_names[0] if len(model_names) > 0 else ""
                    )
                    model_info_box = gr.Textbox(value=models[0]["desc"], lines=3, label=f"{models[0]['platform']}", interactive=False, elem_id="model_info")
                    def model_change(model_name, state):
                        result = next((item for item in models if item["name"] == model_name), None)
                        try:
                            state["model"] = None
                        except Exception as e:
                            pass # No action needed
                        return gr.Textbox(value=result["desc"], label=f"{result['platform']}"), gr.Textbox(autofocus=True), state
                clear = gr.Button(value="Clear history")
                load_pm = gr.Button(value=PRODUCT_BUTTON_EN)
                logo_box = gr.Markdown(value=LOGO_MARKDOWN, elem_id="logo_box")     
        model_selector.change(model_change, [model_selector, state], [model_info_box, msg_box, state])
        msg_box.submit(user, [state, msg_box, chatbot], [msg_box, chatbot], queue=False).then(
            call_llm, inputs=[state, chatbot, model_selector], outputs=[chatbot]
        )
        clear.click(clear_data, [state], [chatbot, state], queue=False)
        load_pm.click(disable_button, inputs=[], outputs=load_pm).then(load_products, inputs=[state], outputs=load_pm)
    return chat_view    

def load_products(state: dict)->gr.Button:
    product_filter = 'APJPIL'
    s4_data = {
        "s4_user": os.environ.get("S4USER"),
        "s4_pass": os.environ.get("S4PASSWORD"),
        "s4_baseurl": os.environ.get("BASE_URL")
    }
    # Get the product data as documents
    msg = "Request to pull product data from S/4HANA Cloud. This might take 2-3 mins."
    logging.info(msg)
    gr.Info(message=msg)
    product_data_docs = get_product_data(
        base_url=s4_data.get("s4_baseurl"), product_filter=product_filter, 
        user=s4_data.get("s4_user"), password=s4_data.get("s4_pass"), 
        max_products=None
    )
    msg = f"Retrieved {len(product_data_docs)} products from S/4HANA Cloud."
    logging.info(msg)
    gr.Info(message=msg)
    # Embed product data from S/4HANA Cloud
    hana_conn = get_hana_connection(conn_params=state["conn_data"])
    status_ok = embed_product_data(conn=hana_conn, embed=DEFAULT_EF, s4data=product_data_docs)
    if status_ok:
        msg = "Completed product embedding to HANA Vector Store."
        gr.Info(message=msg)
        logging.info(msg)
    else:
        msg = "An error occured."
        gr.Warning(message=msg)
        logging.error(msg)
    import time
    time.sleep(2)
    return gr.Button(value=PRODUCT_BUTTON_EN, interactive=True)

def main()->None:
    args = {}
    args["host"] = os.environ.get("HOSTNAME","0.0.0.0")
    args["port"] = os.environ.get("HOSTPORT",51030)
    log_level = int(os.environ.get("APPLOGLEVEL", logging.ERROR))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level,)
    
    logging.info("Connecting to SAP GenAI Hub...")
    
    hana_cloud = {
        "host": os.getenv("HOST"),
        "user": os.getenv("USERNAME",""),
        "password": os.getenv("PASSWORD","") 
    }
    set_proxy_version('gen-ai-hub') # for an AI Core proxy
    global GENAIHUB_PROXY_CLIENT
    GENAIHUB_PROXY_CLIENT = GenAIHubProxyClient()

    # Creat chat UI
    chat_view = build_chat_view(models=MODELS, conn_data=hana_cloud)
    chat_view.queue(max_size=10)
    chat_view.launch(
        debug=False,
        show_api=False,
        server_name=args["host"],
        server_port=args["port"],
        allowed_paths=["./img"]
    )
    
if __name__ == "__main__":
    main()
