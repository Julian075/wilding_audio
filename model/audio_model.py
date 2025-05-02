import torch
import torch.nn.functional as F

class CALMAttention(torch.nn.Module):
    def __init__(self, dim_emb, num_classes,k_shots=1):
        super(CALMAttention, self).__init__()
        self.k_shots=k_shots # Number of shots for each class
        self.key_layer = torch.nn.Linear(dim_emb, dim_emb, bias=True)  # Shared linear layer
        self.query_layer = torch.nn.Linear(dim_emb, dim_emb, bias=True)  # Shared linear layer
        self.num_classes = num_classes
        # Initialize key_layer weights as an identity matrix
        #with torch.no_grad():
        #    self.query_key_layer.weight.copy_(torch.eye(dim_emb))


    def phi_function(self,temperature, x):
        # Apply the scaling function ϕ(x) = exp(b * (1 - x))
        return torch.exp(temperature * (1 - x))

    def forward(self, query, key,onehot,temperature):
        # query: (batch_size, dim_emb)
        # key: (num_classes * num_shots, dim_emb)

        # Step 1: Normalize the query and key
        #query = query.squeeze(1)  # (batch_size, dim_emb)
        #key = key.squeeze(1)  # (Num_clases, dim_emb)
        if len(query.shape)<2:
            query = query.unsqueeze(0)
        query = F.normalize(query,p=2,dim=-1) # (batch_size, dim_emb) dim_emb=512 en biolingual
        key = F.normalize(key,p=2,dim=-1)     # (Num_clases x k_shots, dim_emb) dim_emb=512 en biolingual

        # Step 2: Linear projections for query, key, and value
        Q = self.query_layer(query)  # Projected query
        K = self.key_layer(key)    # Projected key
        #V =  np.repeat(np.eye(self.num_classes), self.k_shots, axis=0) # ( Num_clases x k_shots , Num_clases)
        V = onehot #torch.tensor(V)
        O =[]
        for sample in Q :
            o_i=0
            for  k_shot,one_hot_vector in zip(K,V) :
                    s_i_j=self.phi_function(temperature,sample.t()@k_shot)
                    o_i = o_i + s_i_j*one_hot_vector
            O.append(o_i)



        # Step 3: Compute affinity matrix
       # scores = Q @ K.t() #(batch_size, dim_emb) @ (dim_emb , Num_clases x k_shots) =  scores (batch_size , Num_clases x k_shots)

        # Step 4: Apply the custom ϕ function for the scale
        #affinity = self.phi_function(scores)  #(batch_size , Num_clases x k_shots)
        # Step 5: Calculate the weighted sum output of all classes
        #attention_output =  affinity @ V  # (batch_size , Num_clases x k_shots) @ ( Num_clases x k_shots , Num_clases) =  (batch_size,Num_clases)

        return torch.stack(O).squeeze()

class Audio_classifier(torch.nn.Module):
    def __init__(self,nclasses,k_shots,embed_size, support_audio):
        super(Audio_classifier, self).__init__()
        # parameter to add the two similarity matrix
        self.AttentionLayer = CALMAttention(embed_size, nclasses ,k_shots=k_shots)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #calculate the audio embeddings for the support audios
        # Unpacking a batch of tuples
        self.one_hot_vectors = torch.stack([ torch.tensor(item[1][1]) for item in support_audio]).to(self.device)
        self.one_hot_vectors = self.one_hot_vectors.squeeze()
        self.support_audio_emb = torch.stack([ torch.tensor(item[1][0])for item in support_audio]).to(self.device)
        self.support_audio_emb = self.support_audio_emb.squeeze()









    def forward(self, audio_emb_test,prompts_test,temperature,alpha):

        ## Normalize
        if len(audio_emb_test.shape) >=2:
            audio_embeddings = F.normalize(audio_emb_test, p=2, dim=1)
        elif len(audio_emb_test.shape) ==1:
            audio_embeddings = F.normalize(audio_emb_test.unsqueeze(0) , p=2, dim=1)
        text_embeddings = F.normalize(prompts_test, p=2, dim=1)

        ## Compute cosine similarity
        similarity_matrix_ZSL = audio_embeddings @ text_embeddings.T

        #PFL
        similarity_matrix_PSL=self.AttentionLayer(audio_emb_test,self.support_audio_emb,self.one_hot_vectors,temperature)
        similarity_matrix_PSL = similarity_matrix_PSL / similarity_matrix_PSL.norm(dim=-1, keepdim=True)
        #similarity_matrix_ZSL = torch.norm(similarity_matrix_ZSL, p=2, dim=1, keepdim=True)
        #similarity_matrix_PSL = torch.norm(similarity_matrix_PSL, p=2, dim=1, keepdim=True)

        final_similarity_matrix=((1-alpha)*similarity_matrix_ZSL) + (alpha * similarity_matrix_PSL )

        out_logits = final_similarity_matrix / final_similarity_matrix.norm(dim=-1, keepdim=True)



        return out_logits

    def Contrastive_loss (self, logits: torch.Tensor,label,t) :
        loss_i=0
        for b in range(len(logits)):
            num = torch.exp(logits[b][label[b]]/t)
            dem = torch.sum(torch.exp(logits[b]/t))
            loss_i+=torch.log(num/dem)
        loss=-loss_i/len(logits)
        return loss




