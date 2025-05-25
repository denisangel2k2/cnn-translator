import streamlit as st
import torch
import spacy
import json
import os 


st.set_page_config(page_title="NMT App", layout="centered")

st.title("🌐 English-Romanian Neural Machine Translation")
st.markdown("Enter an English sentence below to get its Romanian translation using a Seq2Seq model with CNN Encoder and LSTM Luong Attention.")


# --- Funcție pentru Încărcare Resurse (cu @st.cache_resource) ---
@st.cache_resource
def load_resources():
    """
    Încarcă toate resursele necesare pentru aplicație:
    modelele SpaCy, vocabularul și modelele TorchScript.
    """
    st.write("Loading resources... This might take a moment.")
    
    # Detectează dispozitivul disponibil (CUDA, MPS sau CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    st.write(f"Using device: **{device}**")

    # Încarcă modelele SpaCy pentru tokenizare
    try:
        en_nlp = spacy.load("en_core_web_sm")
        ro_nlp = spacy.load("ro_core_news_sm")
    except OSError:
        st.error("SpaCy models 'en_core_web_sm' or 'ro_core_news_sm' not found.")
        st.info("Please run: `python -m spacy download en_core_web_sm` and `python -m spacy download ro_core_news_sm` in your terminal.")
        st.stop() # Oprește execuția aplicației dacă modelele SpaCy lipsesc

    # Verifică și încarcă vocabularul sursă (engleză)
    source_vocab_path = "traced_model/source_vocab.json"
    if not os.path.exists(source_vocab_path):
        st.error(f"Error: {source_vocab_path} not found. Please ensure you've run the model saving script and placed files in 'traced_model' directory.")
        st.stop()
    with open(source_vocab_path, "r") as f:
        source_vocab_data = json.load(f)
    
    # Verifică și încarcă vocabularul țintă (română)
    target_vocab_path = "traced_model/target_vocab.json"
    if not os.path.exists(target_vocab_path):
        st.error(f"Error: {target_vocab_path} not found. Please ensure you've run the model saving script and placed files in 'traced_model' directory.")
        st.stop()
    with open(target_vocab_path, "r") as f:
        target_vocab_data = json.load(f)

    # Verifică și încarcă encoderul trasat (TorchScript)
    encoder_path = "traced_model/traced_cnn_encoder.pt"
    if not os.path.exists(encoder_path):
        st.error(f"Error: {encoder_path} not found. Please ensure you've run the model saving script and placed files in 'traced_model' directory.")
        st.stop()
    traced_encoder = torch.jit.load(encoder_path, map_location=device)
    traced_encoder.eval() 

    decoder_path = "traced_model/traced_decoder_with_attention.pt"
    if not os.path.exists(decoder_path):
        st.error(f"Error: {decoder_path} not found. Please ensure you've run the model saving script and placed files in 'traced_model' directory.")
        st.stop()
    traced_decoder = torch.jit.load(decoder_path, map_location=device)
    traced_decoder.eval() 

    st.success("Resources loaded successfully!")

    return {
        "device": device,
        "en_nlp": en_nlp,
        "source_stoi": source_vocab_data["stoi"],
        "source_itos": source_vocab_data["itos"],
        "source_init_token": source_vocab_data["init_token"],
        "source_eos_token": source_vocab_data["eos_token"],
        "source_unk_token": source_vocab_data["unk_token"],
        "target_stoi": target_vocab_data["stoi"],
        "target_itos": target_vocab_data["itos"],
        "target_init_token": target_vocab_data["init_token"],
        "target_eos_token": target_vocab_data["eos_token"],
        "target_unk_token": target_vocab_data["unk_token"],
        "traced_encoder": traced_encoder,
        "traced_decoder": traced_decoder
    }


resources = load_resources()


device = resources["device"]
en_nlp = resources["en_nlp"]
source_stoi = resources["source_stoi"]
source_itos = resources["source_itos"]
source_init_token = resources["source_init_token"]
source_eos_token = resources["source_eos_token"]
source_unk_token = resources["source_unk_token"]
target_stoi = resources["target_stoi"]
target_itos = resources["target_itos"]
target_init_token = resources["target_init_token"]
target_eos_token = resources["target_eos_token"]
target_unk_token = resources["target_unk_token"]
traced_encoder = resources["traced_encoder"]
traced_decoder = resources["traced_decoder"]


def translate_sentence_streamlit(sentence, encoder, decoder, en_nlp, 
                                 source_stoi, source_itos, source_init_token, source_eos_token, source_unk_token,
                                 target_stoi, target_itos, target_init_token, target_eos_token,
                                 device, max_len=50):

    encoder.eval() 
    decoder.eval()

    tokens = [token.text.lower() for token in en_nlp(sentence)]
    tokens = [source_init_token] + tokens + [source_eos_token]
    
    src_indexes = [source_stoi.get(token, source_stoi[source_unk_token]) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_output_tuple = encoder(src_tensor)
        
        if len(encoder_output_tuple) == 2:
            hidden_cell_pair = encoder_output_tuple[0] 
            encoder_outputs = encoder_output_tuple[1]  
            
            hidden_raw = hidden_cell_pair[0]
            cell_raw = hidden_cell_pair[1]
        else:
            st.error(f"Unexpected encoder output structure. Expected 2 elements, got {len(encoder_output_tuple)}. Please check the traced encoder output.")
            if len(encoder_output_tuple) == 3: 
                hidden_raw = encoder_output_tuple[0]
                cell_raw = encoder_output_tuple[1]
                encoder_outputs = encoder_output_tuple[2]
            else:
                return ["ERROR_ENCODER_OUTPUT_STRUCTURE"]


        hidden = hidden_raw[0] if isinstance(hidden_raw, tuple) and len(hidden_raw) == 1 else hidden_raw
        cell = cell_raw[0] if isinstance(cell_raw, tuple) and len(cell_raw) == 1 else cell_raw
        
        if isinstance(encoder_outputs, tuple) and len(encoder_outputs) == 1:
            encoder_outputs = encoder_outputs[0]

    trg_indexes = [target_stoi[target_init_token]]
    
    for _ in range(max_len): 
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        decoder_output_tuple = decoder(trg_tensor, hidden, cell, encoder_outputs)
        output = decoder_output_tuple[0]
        hidden = decoder_output_tuple[1]
        cell = decoder_output_tuple[2]

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == target_stoi[target_eos_token]:
            break


    trg_tokens = [target_itos[i] for i in trg_indexes]
    
    if trg_tokens and trg_tokens[0] == target_init_token:
        trg_tokens = trg_tokens[1:]
    if trg_tokens and trg_tokens[-1] == target_eos_token:
        trg_tokens = trg_tokens[:-1]

    return trg_tokens



input_sentence = st.text_area("Enter English Sentence:", height=100, placeholder="e.g., Hello, how are you?")


if st.button("Translate"):
    if input_sentence.strip() == "":
        st.warning("Please enter a sentence to translate.")
    else:
        with st.spinner("Translating..."): 
            translated_tokens = translate_sentence_streamlit(
                input_sentence,
                traced_encoder,
                traced_decoder,
                en_nlp,
                source_stoi, source_itos, source_init_token, source_eos_token, source_unk_token,
                target_stoi, target_itos, target_init_token, target_eos_token,
                device
            )
            st.success("Translation Complete!")
            st.write("### Translated Romanian Sentence:")
            st.info(" ".join(translated_tokens))

st.markdown("---")
st.markdown("### About this App:")
st.markdown("""
This application demonstrates a Neural Machine Translation (NMT) model. 
It uses a **Convolutional Neural Network (CNN) based Encoder** to process the input English sentence 
and an **LSTM-based Decoder with a Luong Concatenation Attention Mechanism** to generate the Romanian translation. 
The model components are saved and loaded using **TorchScript** for efficient deployment.
""")