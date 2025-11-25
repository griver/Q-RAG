from torch.utils.data import Dataset
import envs.chunker

CHUNK_SIZE = 2000



class QADatasetAdapter(Dataset):
    """
    Simple adapter that adapts datasets Babilong, HotPotQA and MUSIQUE for QARetrievalEnv.
    This adapter doesn't tokenize or embeds text chunks.

    You can create different adapter that for example tokenize every text in a sample or
    build faiss index over text chunks.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = self.dataset.name()
        #print(f"{self.dataset_name} dataset length: {self.dataset.__len__()}")


    def __getitem__(self, index):
        sample = self.dataset[index]
        sf_idx = []
        chunks_texts = []

        if self.dataset_name == "combined":
            source = sample.get('source')
            if source not in ('hotpotqa', 'musique', 'babilong'):
                raise ValueError(f"Invalid or missing 'source' in combined dataset sample: {source}")
        else:
            source = self.dataset_name

        if source == 'hotpotqa':
            sp_title_set = set()
            sample_id = sample['_id']
            question = sample["question"]
            answer = sample["answer"]
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])
            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                chunk = title + " " + " ".join(sentences)
                chunks_texts.append(chunk)

        elif source == 'musique':
            sample_id = sample['id']
            question = sample["question"]
            answer = sample["answer"]
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                chunk = para['title'] + '. ' + para['paragraph_text']
                chunks_texts.append(chunk)
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])

        elif source == 'babilong':
            sample_id = index
            question = sample["question"]
            answer = sample["answer"]
            chunks_texts = sample['chunks']
            sf_idx = list(sample['references_idx'])

        elif source == "longbench":
            sample_id = sample["_id"]
            question = sample["input"]
            answer = sample["answers"][0]
            chunks_texts = envs.chunker.chunks_split(sample["context"], chunk_size=CHUNK_SIZE)
            #sf_idx = None
            sf_idx.append(0)
            #print("Chunks count:", len(chunks_texts))

        elif source == "gsm8k":
            sample_id = index
            question = sample["question"]
            answer = envs.dataloaders.gsm8k.extract_short_answer(sample["answer"])
            chunks_texts = self.dataset.get_examples()
            sf_idx = [0]

        else:
            raise ValueError(f"Unsupported dataset/source: {source}")

        #if question.endswith("?"):  question = question[:-1]
        result = {
            'id': sample_id,
            'question': question,
            'answer': answer, # sample["answer"],
            'chunks': chunks_texts,
            'sf_idx': sf_idx,
        }
        # if len(chunks_texts) != 10:
        #     print(f'sample {sample_id}, num_chunks: {len(chunks_texts)}')
        #     print('Q:', question)
        #     print('A:', sample['answer'])
        #     print('sf_idx:', result['sf_idx'])
        #     for i, ch in enumerate(chunks_texts):
        #         print(f"== CH#{i} ==\n {ch}")
        return result


    def __len__(self):
        return self.dataset.__len__()
