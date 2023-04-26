# GPT-NER: Named Entity Recognition via Large LanguageModels

## Introduction
This repo contains code for the paper [GPT-NER: Named Entity Recognition via Large LanguageModels](https://arxiv.org/pdf/2304.10428.pdf).
```latex
@article{wang2023gpt,
  title={GPT-NER: Named Entity Recognition via Large Language Models},
  author={Wang, Shuhe and Sun, Xiaofei and Li, Xiaoya and Ouyang, Rongbin and Wu, Fei and Zhang, Tianwei and Li, Jiwei and Wang, Guoyin},
  journal={arXiv preprint arXiv:2304.10428},
  year={2023}
}
```

## Usage

### Requirements

* python>=3.7.3
* openai==0.27.2
* simcse==0.4

This repor mainly use two addtional packages: [SimCSE](https://github.com/princeton-nlp/SimCSE) and [OpenAI](https://github.com/openai/openai-python). So, if you want to know more about the arguments used in codes, please refer to the corresponding documents.

### Proposed Dataset
For the full NER dataset, we follow [MRC-NER](https://arxiv.org/pdf/1910.11476.pdf) for preprocessing, and you can directly download these [here](https://github.com/ShannonAI/mrc-for-flat-nested-ner).

For sampled **100-dataset**, we have put them on the [Google Drive](https://drive.google.com/drive/folders/1ByoM4Bb_BRmvp_D28QGRxrxfxnQtYeQi?usp=share_link).

### Few-shot Demonstrations Retrieval
For sentence-level embeddings, run `openai_access/extract_mrc_knn.py`.

Note that you should change the directory for the input/output file and the used SimCSE model. In this repo, the model `sup-simcse-roberta-large` is used for SimCSE, and you can find it [here](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large).

### OpenAI Access

We follow the official steps to access the GPT-* Models, and the document can be found [here](https://platform.openai.com/docs/api-reference/introduction). Before you run our scripts, you need to add **OPENAI_API_KEY**, which you can find it in your account profile, to the environment variable by the command `export OPENAI_API_KEY="YOUR_KEY"`.

To get preditions, please run `openai_access/scripts/access_ai.sh`, and the used arguments are listed in file `openai_access/get_results_mrc_knn.py`.

For self-verification, please run `openai_access/scripts/verify.sh`, and the used arguments are listed in file `openai_access/verify_results.py`.

**Note that accessing to the `GPT-3` is very expensive, we thus strongly advise you to start from our sampled 100-dataset.**

### Evaluate

We use span-level precession, recall and F1-score for evaluation, and to do this, please run the script `openai_access/scripts/compute_f1.sh`.

## Results

Table 1: Results of sampled 100 pieces of data for two **Flat** NER datasets: CoNLL2003 and OntoNotes5.0.
<table border=2>
    <tr>
        <td></td>
        <td align="center" colspan="3"><b> EnglishCoNLL2003 (Sampled 100) </b> </td>
        <td align="center" colspan="3"><b> EnglishOntoNotes5.0 (Sampled 100) </b> </td>
    </tr>
    <tr>
        <td>Model</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
    </tr>
    <tr>
        <td align="center" colspan="7"> Baselines (Supervised Model) </td> 
    </tr>
    <tr>
        <td>ACE+document-context</td> 
        <td>97.8</td> 
        <td>98.28</td> 
        <td><b>98.04 (SOTA)</b></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr> 
    <tr>
        <td>BERT-MRC+DSC</td> 
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>93.81</td> 
        <td>93.95</td> 
        <td><b>93.88 (SOTA)</b></td>
    </tr>
    <tr>
        <td align="center" colspan="7"> GPT-NER </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>88.18</td> 
        <td>78.54</td> 
        <td>83.08</td>
        <td>64.21</td> 
        <td>65.51</td> 
        <td>64.86</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>90.47</td> 
        <td>95</td> 
        <td>92.68</td>
        <td>76.08</td> 
        <td>83.06</td> 
        <td>79.57</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>94.06</td> 
        <td>96.54</td> 
        <td>95.3</td>
        <td>78.38</td> 
        <td>83.9</td> 
        <td>81.14</td>
    </tr> 
    <tr>
        <td align="center" colspan="7"> Self-verification (zero-shot) </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>88.95</td> 
        <td>79.73</td> 
        <td>84.34</td>
        <td>64.94</td> 
        <td>65.90</td> 
        <td>65.42</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>91.77</td> 
        <td>96.36</td> 
        <td>94.01</td>
        <td>77.33</td> 
        <td>83.29</td> 
        <td>80.31</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>94.15</td> 
        <td>96.77</td> 
        <td>95.46</td>
        <td>79.05</td> 
        <td>83.71</td> 
        <td>81.38</td>
    </tr> 
    <tr>
        <td align="center" colspan="7"> Self-verification (few-shot) </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>90.04</td> 
        <td>80.14</td> 
        <td>85.09</td>
        <td>65.21</td> 
        <td>66.25</td> 
        <td>65.73</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>92.92</td> 
        <td>95.45</td> 
        <td>94.17</td>
        <td>77.64</td> 
        <td>83.22</td> 
        <td>80.43</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>94.73</td> 
        <td>96.97</td> 
        <td>95.85</td>
        <td>79.25</td> 
        <td>83.73</td> 
        <td>81.49</td>
    </tr> 
</table>

Table 2: Results of full data for two Flat NER datasets: CoNLL2003 and OntoNotes5.0.
<table border=2>
    <tr>
        <td></td>
        <td align="center" colspan="3"><b> English CoNLL2003 (FULL) </b> </td>
        <td align="center" colspan="3"><b> English OntoNotes5.0 (FULL) </b> </td>
    </tr>
    <tr>
        <td>Model</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
    </tr>
    <tr>
        <td align="center" colspan="7"> Baselines (Supervised Model) </td> 
    </tr>
    <tr>
        <td>BERT-Tagger</td> 
        <td>-</td> 
        <td>-</td> 
        <td>92.8</td>
        <td>90.01</td>
        <td>88.35</td>
        <td>89.16</td>
    </tr>
    <tr> 
        <td>BERT-MRC</td> 
        <td>92.33</td>
        <td>94.61</td>
        <td>93.04</td>
        <td>92.98</td> 
        <td>89.95</td> 
        <td>91.11</td>
    </tr>
    <tr>
        <td>GNN-SL</td> 
        <td>93.02</td> 
        <td>93.40</td> 
        <td>93.2</td>
        <td>91.48</td>
        <td>91.29</td>
        <td>91.39</td>
    </tr>
    <tr> 
        <td>ACE+document-context</td> 
        <td>-</td>
        <td>-</td>
        <td><b>94.6 (SOTA)</b></td>
        <td>-</td> 
        <td>-</td> 
        <td>-</td>
    </tr>
    <tr> 
        <td>BERT-MRC+DSC</td> 
        <td>93.41</td>
        <td>93.25</td>
        <td>93.33</td>
        <td>91.59</td> 
        <td>92.56</td> 
        <td><b>92.07 (SOTA)</b></td>
    </tr>
    <tr>
        <td align="center" colspan="7"> GPT-NER </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>77.04</td> 
        <td>68.69</td> 
        <td>72.62</td>
        <td>53.8</td> 
        <td>59.36</td> 
        <td>56.58</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>81.04</td> 
        <td>88.00</td> 
        <td>84.36</td>
        <td>66.87</td> 
        <td>73.77</td> 
        <td>70.32</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>88.54</td> 
        <td>91.4</td> 
        <td>89.97</td>
        <td>74.17</td> 
        <td>79.29</td> 
        <td>76.73</td>
    </tr> 
    <tr>
        <td align="center" colspan="7"> Self-verification (zero-shot) </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>77.13</td> 
        <td>69.23</td> 
        <td>73.18</td>
        <td>54.14</td> 
        <td>59.44</td> 
        <td>56.79</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>83.31</td> 
        <td>88.11</td> 
        <td>85.71</td>
        <td>67.29</td> 
        <td>73.81</td> 
        <td>70.55</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>89.47</td> 
        <td>91.77</td> 
        <td>90.62</td>
        <td>74.64</td> 
        <td>79.52</td> 
        <td>77.08</td>
    </tr> 
    <tr>
        <td align="center" colspan="7"> Self-verification (few-shot) </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>77.50</td> 
        <td>69.38</td> 
        <td>73.44</td>
        <td>54.23</td> 
        <td>59.65</td> 
        <td>56.94</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>83.73</td> 
        <td>88.07</td> 
        <td>85.9</td>
        <td>67.35</td> 
        <td>73.79</td> 
        <td>70.57</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>89.76</td> 
        <td>92.06</td> 
        <td>90.91</td>
        <td>74.89</td> 
        <td>79.51</td> 
        <td>77.20</td>
    </tr> 
</table>

Table 3: Results of full data for three Nested NER datasets: ACE2004, ACE2005 and GENIA.
<table border=2>
    <tr>
        <td></td>
        <td align="center" colspan="3"><b> English ACE2004 (FULL) </b> </td>
        <td align="center" colspan="3"><b> English ACE2005 (FULL) </b> </td>
        <td align="center" colspan="3"><b> English GENIA (FULL) </b> </td>
    </tr>
    <tr>
        <td>Model</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>F1</td>
    </tr>
    <tr>
        <td align="center" colspan="10"> Baselines (Supervised Model) </td> 
    </tr>
    <tr>
        <td>BERT-MRC</td> 
        <td>85.05</td> 
        <td>86.32</td> 
        <td>85.98</td>
        <td>87.16</td>
        <td>86.59</td>
        <td>86.88</td>
        <td>85.18</td>
        <td>81.12</td>
        <td><b>83.75 (SOTA)</b></td>
    </tr>
    <tr>
        <td>Triaffine+BERT</td> 
        <td>87.13</td> 
        <td>87.68</td> 
        <td>87.40</td>
        <td>86.70</td>
        <td>86.94</td>
        <td>86.82</td>
        <td>80.42</td>
        <td>82.06</td>
        <td>81.23</td>
    </tr>
    <tr>
        <td>Triaffine+ALBERT</td> 
        <td>88.88</td> 
        <td>88.24</td> 
        <td>88.56</td>
        <td>87.39</td>
        <td>90.31</td>
        <td>88.83</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>BINDER</td> 
        <td>88.3</td> 
        <td>89.1</td> 
        <td><b>88.7 (SOTA)</b></td>
        <td>89.1</td>
        <td>89.8</td>
        <td><b>89.5 (SOTA)</b></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td align="center" colspan="10"> GPT-NER </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>55.04</td> 
        <td>41.76</td> 
        <td>48.4</td>
        <td>44.5</td> 
        <td>46.24</td> 
        <td>45.37</td>
        <td>44.1</td> 
        <td>38.64</td> 
        <td>41.37</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>65.31</td> 
        <td>53.67</td> 
        <td>60.68</td>
        <td>58.04</td> 
        <td>58.97</td> 
        <td>58.50</td>
        <td>63.43</td> 
        <td>44.17</td> 
        <td>51.68</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>72.23</td> 
        <td>75.01</td> 
        <td>73.62</td>
        <td>71.72</td> 
        <td>74.2</td> 
        <td>73.96</td>
        <td>61.38</td> 
        <td>66.74</td> 
        <td>64.06</td>
    </tr> 
    <tr>
        <td align="center" colspan="10"> Self-verification (zero-shot) </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>55.44</td> 
        <td>42.22</td> 
        <td>48.83</td>
        <td>45.06</td> 
        <td>46.62</td> 
        <td>45.84</td>
        <td>44.31</td> 
        <td>38.79</td> 
        <td>41.55</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>69.64</td> 
        <td>54.98</td> 
        <td>62.31</td>
        <td>59.49</td> 
        <td>60.17</td> 
        <td>59.83</td>
        <td>59.54</td> 
        <td>44.26</td> 
        <td>51.9</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>73.58</td> 
        <td>74.74</td> 
        <td>74.16</td>
        <td>72.63</td> 
        <td>75.39</td> 
        <td>73.46</td>
        <td>61.77</td> 
        <td>66.81</td> 
        <td>64.29</td>
    </tr> 
    <tr>
        <td align="center" colspan="10"> Self-verification (few-shot) </td> 
    </tr>
    <tr>
        <td>GPT-3 + random retrieval</td> 
        <td>55.63</td> 
        <td>42.49</td> 
        <td>49.06</td>
        <td>45.49</td> 
        <td>46.73</td> 
        <td>46.11</td>
        <td>44.68</td> 
        <td>38.98</td> 
        <td>41.83</td>
    </tr> 
    <tr>
        <td>GPT-3 + sentence-level embedding</td> 
        <td>70.17</td> 
        <td>54.87</td> 
        <td>62.52</td>
        <td>59.69</td> 
        <td>60.35</td> 
        <td>60.02</td>
        <td>59.87</td> 
        <td>44.39</td> 
        <td>52.13</td>
    </tr> 
    <tr>
        <td>GPT-3 + entity-level embedding</td> 
        <td>73.29</td> 
        <td>75.11</td> 
        <td>74.2</td>
        <td>72.77</td> 
        <td>75.51</td> 
        <td>73.59</td>
        <td>61.89</td> 
        <td>66.95</td> 
        <td>64.42</td>
    </tr> 
</table>


## Contact
If you have any issues or questions about this repo, feel free to contact wangshuhe@stu.pku.edu.cn.