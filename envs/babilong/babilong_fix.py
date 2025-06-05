import numpy as np
from sympy.stats import sample_stochastic_process
from tqdm import tqdm
from .babilong_utils import TaskDataset, Dataset

class QA2FixWrapper(Dataset):

    def __init__(self, dataset, add_sentence_idx=False):
        super().__init__()
        self.dataset = dataset
        self.dropping_verbs = set(["down", "dropped", "discarded", 'left'])
        self.location_names = set(sample['answer'] for sample in dataset)
        self.add_sentence_idx = add_sentence_idx

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, ind):
        sample = self.dataset.__getitem__(ind)
        return self.process_sample(sample)

    def process_sample(self, sample):
        fix = self.compute_references(sample)
        if self.add_sentence_idx:
            self.add_id_to_facts(sample)
            sample['references'] = [sample['facts'][idx] for idx in fix['references_idx']]

        assert set(fix['references_idx']) == set(sample['references_idx']), 'they should be the same with exception of order'
        sample['references_idx'] = fix['references_idx']
        return sample

    def add_id_to_facts(self, sample):
        sample['facts'] = [f"Fact #{i}: {f}" for i, f in enumerate(sample['facts'])]

    def compute_references(self, sample):
        #sample = dataset[id]
        facts = sample['facts']

        item = self._get_search_item(sample['question'])
        mention_id = self._last_mention(facts, item)
        person = self._get_person(facts[mention_id])
        if self._item_is_dropped(facts[mention_id]):
             location_id = self._last_loc(facts[:mention_id], person)
        else:
             location_id = self._last_loc(facts, person)
        #     location_id = mention_id + self._last_loc(facts[mention_id:], person)

        references_idx = [mention_id, location_id]
        references = [facts[i] for i in references_idx]

        return dict(
            references=references,
            references_idx=references_idx,
            location=self._get_location(references[-1])
        )

    def test_dataset(self, visualize='none'):
        """
        Args:
            * visualize: if visualize == 'all' then visualize all samples.
                         if visualize == 'errors' then visualize only samples where computed answer
                         or reference facts are different from original
                         if visualize == 'none' then nothing is printed
        """
        #self.location_names = self.location_names.union(set(sample['answer'] for sample in dataset))
        for i, sample in tqdm(enumerate(self.dataset)):

            fix = self.compute_references(sample)
            diff = self.check_differences(sample, fix, visualize)
            if  not diff['same_answer']:
                raise ValueError(f'fix gets a wrong answer on sample #{i}')
            elif not diff['same_references']:
                facts_str = '\n'.join(sample['facts'])
                print(f"Question: {sample['question']}")
                print('FACTS:')
                print(facts_str)
                print(f"Answer: {sample['answer']}")
                print('------------')
                print("Dataset Refs:", sample['references'])
                print('Fix Refs:', fix['references'])
                #sample['references'] = fix['references']
                raise ValueError(f'Different references on sample #{i}')

    def check_differences(self, sample, fix, visualize):
        same_references = len(sample['references']) == len(fix['references'])
        if same_references:
            same_references = all(s == f for s,f in zip(sorted(sample['references']), sorted(fix['references'])))

        res = dict(
            same_answer=(sample['answer'] == fix['location']),
            same_references=same_references
        )

        if visualize in [ 'all', 'errors']:
            if visualize == 'all' or not res['same_answer']:
                print('========================================')
                print(f'Q: {sample["question"]}')
                print(f'Answer:\n\tdataset: {sample["answer"]}\n\t fix: {fix["location"]}')

                print('References:')
                print(f'dataset: {sample["references"]}')
                print((f'fix: {fix["references"]}'))

        return res

    def _item_is_dropped(self, sentence):
        for d in self.dropping_verbs:
            if d in sentence:
                return True
        else:
            return False

    def _last_mention(self, facts, item):
        for i in reversed(range(len(facts))):
            if item in facts[i]:
                return i
        else:
            facts_str = "\n".join(facts)
            raise ValueError(f"Can't find {item} in the facts:\n {facts_str}.")

    def _last_loc(self, facts, person):
        for i in reversed(range(len(facts))):
            if person in facts[i]:
                if self._get_location(facts[i]):
                    return i
        else:
            facts_str = "\n".join(facts)
            raise ValueError(f"Can't find any locations in facts:\n{facts_str}")

    def _get_search_item(self, question):
        question = question.strip()
        assert question[-1] == "?", f"Question doesn't end with question mark: [{question}]"
        search_item = question.strip()[:-1].split()[-1]
        return search_item

    def _get_person(self, fact):
        return fact.split()[0]

    def _get_location(self, fact):
        for l in self.location_names:
            if l in fact:
                return l

        return None