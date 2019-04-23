# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers import UniversalDependenciesMultiLangDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestUniversalDependenciesMultilangDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies_multilang"

    def test_read_from_files_first_pass_true(self):
        reader = UniversalDependenciesMultiLangDatasetReader(is_first_pass_for_vocab=True)
        instances = list(reader.read(str(self.data_path)))

        instance = instances[0]
        fields = instance.fields
        assert fields['metadata'].metadata['lang'] == 'es'
        assert [t.text for t in fields["words"].tokens] == ['Aclarando', 'hacia', 'todo', 'el', 'mundo',
                                                            'Valderrama', 'Y', 'Eduardo', 'Son', 'La', 'Misma',
                                                            'Persona', '.']

        assert fields["pos_tags"].labels == ['VERB', 'ADP', 'DET', 'DET', 'NOUN', 'NOUN', 'CONJ', 'NOUN', 'NOUN',
                                             'DET', 'ADJ', 'NOUN', '.']
        assert fields["head_tags"].labels == ['ROOT', 'adpmod', 'det', 'det', 'adpobj', 'nsubj', 'cc', 'conj', 'xcomp',
                                              'det', 'amod', 'attr', 'p']
        assert fields["head_indices"].labels == [0, 1, 5, 5, 2, 9, 6, 6, 1, 12, 12, 9, 1]

        instance = instances[1]
        fields = instance.fields
        assert fields['metadata'].metadata['lang'] == 'es'
        assert [t.text for t in fields["words"].tokens] == ['Es', 'un', 'bar', 'disfrazado', 'de', 'restaurante', 'la',
                                                            'comida', 'esta', 'demasiado', 'salada', '.']
        assert fields["pos_tags"].labels == ['VERB', 'DET', 'NOUN', 'VERB', 'ADP', 'NOUN',
                                             'DET', 'NOUN', 'VERB', 'PRON', 'ADJ', '.']
        assert fields["head_tags"].labels == ['ROOT', 'det', 'attr', 'partmod', 'adpmod', 'adpobj',
                                              'det', 'nsubj', 'parataxis', 'nmod', 'acomp', 'p']
        assert fields["head_indices"].labels == [0, 3, 1, 3, 4, 5, 8, 9, 1, 11, 9, 1]

        instance = instances[2]
        fields = instance.fields
        assert fields['metadata'].metadata['lang'] == 'fr'
        assert [t.text for t in fields["words"].tokens] == ['Ses', 'habitants', 'sont', 'appelés', 'les', 'Paydrets',
                                                            'et', 'les', 'Paydrètes', ';']
        assert fields["pos_tags"].labels == ['DET', 'NOUN', 'VERB', 'VERB', 'DET',
                                             'NOUN', 'CONJ', 'DET', 'NOUN', '.']
        assert fields["head_tags"].labels == ['poss', 'nsubjpass', 'auxpass', 'ROOT', 'det', 'attr'
                                              'cc', 'det', 'conj', 'p']
        assert fields["head_indices"].labels == [2, 4, 4, 0, 6, 4, 6, 9, 6, 4]

        instance = instances[3]
        fields = instance.fields
        assert fields['metadata'].metadata['lang'] == 'fr'
        assert [t.text for t in fields["words"].tokens] == ['Cette', 'tour', 'de', 'a',
                                                            'été', 'achevée', 'en', '1962', '.']
        assert fields["pos_tags"].labels == ['DET', 'NOUN', 'ADP', 'VERB', 'VERB',
                                             'VERB', 'ADP', 'NUM', '.']
        assert fields["head_tags"].labels == ['det', 'nsubjpass', 'adpmod', 'aux', 'auxpass', 'ROOT',
                                              'adpmod', 'adpobj', 'p']
        assert fields["head_indices"].labels == [2, 6, 2, 6, 6, 0, 6, 7, 6]

        instance = instances[4]
        fields = instance.fields
        assert fields['metadata'].metadata['lang'] == 'it'
        assert [t.text for t in fields["words"].tokens] == ['Inconsueto', 'allarme', 'alla', 'Tate', 'Gallery', ':']
        assert fields["pos_tags"].labels == ['ADJ', 'NOUN', 'ADP', 'NOUN', 'NOUN', '.']
        assert fields["head_tags"].labels == ['amod', 'ROOT', 'adpmod', 'dep', 'adpobj', 'p']
        assert fields["head_indices"].labels == [2, 0, 2, 5, 3, 2]

        instance = instances[5]
        fields = instance.fields
        assert fields['metadata'].metadata['lang'] == 'it'
        assert [t.text for t in fields["words"].tokens] == ['Hamad', 'Butt', 'è', 'morto', 'nel', '1994', 'a', '32'
                                                            'anni', '.']
        assert fields["pos_tags"].labels == ['NOUN', 'NOUN', 'VERB', 'VERB', 'ADP',
                                             'NUM', 'ADP', 'NUM', 'NOUN', '.']
        assert fields["head_tags"].labels == ['dep', 'nsubj', 'aux', 'ROOT', 'adpmod', 'adpobj',
                                              'adpmod', 'num', 'adpobj', 'p']
        assert fields["head_indices"].labels == [2, 4, 4, 0, 4, 5, 4, 9, 7, 4]

    def test_read_from_files_first_pass_false(self):
        reader = UniversalDependenciesMultiLangDatasetReader(is_first_pass_for_vocab=False)
        coun_es, coun_fr, coun_it = 0, 0, 0
        for instance in reader.read(str(self.data_path)):
            lang = instance.fields['metadata'].metadata['lang']
            if lang == 'es':
                coun_es += 1
                if coun_es > 2:
                    break
            if lang == 'fr':
                coun_fr += 1
                if coun_fr > 2:
                    break
            if lang == 'it':
                coun_it += 1
                if coun_it > 2:
                    break
        # Asserting that the reader didn't stop after finishing reading the three files
        assert (coun_es > 2 or coun_fr > 2 or coun_it > 2)
