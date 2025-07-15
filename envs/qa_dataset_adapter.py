from torch.utils.data import Dataset

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

    def __getitem__(self, index):
        sample = self.dataset[index]
        question = sample["question"]
        if question.endswith("?"):
            question = question[:-1]

        sf_idx = []
        chunks_texts = []
        sample_id = None
        if self.dataset_name == 'hotpotqa':
            sp_title_set = set()
            sample_id = sample['_id']
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])

            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                chunk = title + " " + " ".join(sentences)
                chunks_texts.append(chunk)

        elif self.dataset_name == 'musique':
            sample_id = sample['id']
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                chunk = para['title'] + '. ' + para['paragraph_text']
                chunks_texts.append(chunk)

            # label order
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])

        elif self.dataset_name == 'babilong':
            sample_id = index
            chunks_texts = sample['chunks']
            sf_idx = list(sample['references_idx'])

        result = {
            'id': sample_id,
            'question': question,
            'answer': sample["answer"],
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