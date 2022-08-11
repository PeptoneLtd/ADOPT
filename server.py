from typing import List
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from adopt import MultiHead, ZScorePred
import uvicorn

MODEL_TYPE = "esm-1b"
STRATEGY = "train_on_cleared_1325_test_on_117_residue_split"

z_score_pred = ZScorePred(STRATEGY, MODEL_TYPE)
multi_head = MultiHead(MODEL_TYPE)


origins = [
    "http://localhost:8000",
    "http://localhost:5173",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Sequence(BaseModel):
    id: str
    name: str
    seq: str

class ZScores(Sequence):
    z_scores: List[float]

class Response(BaseModel):
    z_scores: List[ZScores]

class BulkSequenceRequest(BaseModel):
    sequences: List[Sequence]


# # sequence as path paramenter could make sense for caching requests, but may not be viable for sequences longer than ~2000
# @app.get("/sequence/{sequence}/z_score")
# async def get_z_score(sequence: str):
#     multi_head = MultiHead(MODEL_TYPE, sequence, None)
#     representation, tokens = multi_head.get_representation()
#     predicted_z_scores = [ ZScores(sequence=sequence, z_scores=z_score_pred.get_z_score(representation).tolist()) ]
#     return Response(z_scores=predicted_z_scores)


@app.post("/bulk/z_score")
async def get_bulk_z_score(bulkSequenceRequest: BulkSequenceRequest):
    z_scores: List[ZScores] = []
    for s in bulkSequenceRequest.sequences:
        representation, tokens = multi_head.get_representation(s.seq, s.id)
        predicted_z_scores = z_score_pred.get_z_score(representation).tolist()
        z_scores.append(ZScores(id=s.id, name=s.name, seq=s.seq, z_scores=predicted_z_scores))
    return Response(z_scores=z_scores)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)