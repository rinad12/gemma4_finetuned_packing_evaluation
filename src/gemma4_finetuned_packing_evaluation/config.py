BASE_MODEL_ID = "google/gemma-4-E4B"
ADAPTER_ID    = "rinad12/gemma_4_fine_tuned_packing"
DATASET_ID    = "rinad12/Syn-Item-Lists"

SYSTEM_PROMPT = (
    "You are a professional travel assistant specialized in extreme logistics. "
    "Provide Reasoning followed by the Packing List in JSON format."
    "Structure your response as follows:\n"
    "Reasoning: [your analysis]\n\n"
    "Packing List: [JSON array of objects]"
)

NUM_SAMPLES   = 50
NUM_FEW_SHOT  = 2
MAX_NEW_TOKENS = 3000
MAX_SEQ_LENGTH = 4096
