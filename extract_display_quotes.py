#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jul  4 14:55:46 2022

@author: sjufri

This code has been adapted (with permission) from the GenderGapTracker GitHub page 
(https://github.com/sfu-discourse-lab/GenderGapTracker/tree/master/NLP/main)
and modified to run on a Jupyter Notebook.

The quotation tool’s accuracy rate is evaluated in the below article:
The Gender Gap Tracker: Using Natural Language Processing to measure gender bias in media
(https://doi.org/10.1371/journal.pone.0245533)
'''

# import required packages
import os
import io
import sys
import codecs
import logging
import traceback
from collections import Counter
from datetime import datetime
import hashlib

# matplotlib: visualization tool
from matplotlib import pyplot as plt

# pandas: tools for data processing
import pandas as pd

# spaCy and NLTK: natural language processing tools for working with language/text data
import spacy
from spacy import displacy
from spacy.tokens import Span
import nltk
nltk.download('punkt')
from nltk import Tree
from nltk.tokenize import sent_tokenize

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, Markdown, clear_output

# clone GenderGapTracker GitHub page
path  = './'
clone = 'git clone https://github.com/sfu-discourse-lab/GenderGapTracker'
os.chdir(path)
os.system(clone)

# import the quote extractor tool
from config import config
sys.path.insert(0,'./GenderGapTracker/NLP/main')
from quote_extractor import extract_quotes
import utils


class QuotationTool():
    '''
    Interactive tool for extracting and displaying quotes in a text
    '''
    
    def __init__(self):
        '''
        Initiate the QuotationTool
        '''
        # initiate the app_logger
        self.app_logger = utils.create_logger('quote_extractor', log_dir='logs', 
                                         logger_level=logging.INFO,
                                         file_log_level=logging.INFO)
        
        # download spaCy's en_core_web_lg, the pre-trained English language tool from spaCy
        print('Loading spaCy language model...')
        self.nlp = spacy.load('en_core_web_lg')
        print('Finished loading.')
        
        # initiate variables to hold texts and quotes in pandas dataframes
        self.text_df = None
        self.quotes_df = None
        
        # initiate other required variables
        self.html = None
        self.current_text = None

        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv or xlsx)',
            accept='.txt, .xlsx, .csv ', # accepted file extension 
            multiple=True,  # True to accept multiple files
            layout = widgets.Layout(width='320px')
            )
        
        self.upload_out = widgets.Output()
        
        # give notification when file is uploaded
        def _cb(change):
            with self.upload_out:
                clear_output()
                print('Uploading files...')
                try:
                    self.process_upload(deduplication=True)
                    print('Finished uploading files.')
                    print('Currently {} text documents are loaded for analysis'.format(self.text_df.shape[0]))
                except:
                    print('Please upload your text file in the above cell!')
            
        self.file_uploader.observe(_cb, names='data')
        self.upload_box = widgets.VBox([self.file_uploader, self.upload_out])


    def load_txt(self, value):
        '''
        Load individual txt file content and return a dict object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            value: the file containing the text data
        '''
        temp = {'text_name': value['metadata']['name'][:-4],
                'text': codecs.decode(value['content'], encoding='utf-8')
        }
        
        return [temp]


    def load_table(self, value, file_fmt):
        '''
        Load csv or xlsx file
        
        Args:
            value: the file containing the text data
            file_fmt: the file format, i.e., 'csv', 'xlsx'
        '''
        # read the file based on the file format
        if file_fmt == 'csv':
            temp_df = pd.read_csv(io.BytesIO(value['content']))
        if file_fmt == 'xlsx':
            temp_df = pd.read_excel(io.BytesIO(value['content']))
            
        # Check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) or ('text_name' not in temp_df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(value['metadata']['name']))
            return []
        
        # Return a list of dict objects
        temp = temp_df[['text_name', 'text']].to_dict(orient='index').values()
        
        return temp


    def hash_gen(self, temp_df):
        '''
        Create column text_id by md5 hash of the text in text_df
        
        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        temp_df['text_id'] = temp_df['text'].apply(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest())
        
        return temp_df


    def nlp_preprocess(self, temp_df):
        '''
        Pre-process text and fit it with Spacy language model into the column "spacy_text"

        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        temp_df['spacy_text'] = temp_df['text']\
            .map(sent_tokenize)\
                .apply(lambda t: ' '.join(t))\
                    .map(utils.preprocess_text)\
                        .map(self.nlp)
                        
        return temp_df


    def process_upload(self, deduplication=True):    
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            txt_upload: the uploaded .txt files from upload_files()
        '''
        # create an empty list for a placeholder to store all the texts
        all_data = []
        
        for file in self.file_uploader.value.keys():
            if file.lower().endswith('txt'):
                text_dic = self.load_txt(self.file_uploader.value[file])
            else:
                text_dic = self.load_table(self.file_uploader.value[file], \
                    file_fmt=file.lower().split('.')[-1])
            all_data.extend(text_dic)
        
        uploaded_df = pd.DataFrame.from_dict(all_data)

        uploaded_df = self.hash_gen(uploaded_df)
        uploaded_df = self.nlp_preprocess(uploaded_df)
        self.text_df = pd.concat([self.text_df, uploaded_df])
        self.text_df.reset_index(drop=True, inplace=True)
        
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
    
    
    def extract_inc_ent(self, list_of_string, spacy_doc, inc_ent):
        '''
        Extract included named entities from a list of string

        Args:
            list_of_string: a list of string from which to extract the named entities
            spacy_doc: spaCy's processed text for the above list of string
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''       
        return [
            [(str(ent), ent.label_) for ent in spacy_doc.ents \
                if (str(ent) in string) & (ent.label_ in inc_ent)]\
                    for string in list_of_string
                    ]
        

    def get_quotes(self, inc_ent, create_tree=False):
        '''
        Extract quotes and their meta-data (quote_id, quote_index, etc.) from the text
        and return as a pandas dataframe

        Args:
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
            create_tree: option to create parse tree files for the quotes 
        '''
        # create an output folder if not already exist
        os.makedirs('output', exist_ok=True)
        
        # create a tree folder and specify the file path if create_tree=True
        if create_tree:
            os.makedirs('output/trees', exist_ok=True)
            #tree_dir = './output/trees/'
            OUTPUT_DIRECTORY = './output/trees/'
        else:
            tree_dir = None
    
        # create an empty list to store all detected quotes
        all_quotes = []
        
        # go through all the texts and start extracting quotes
        for row in self.text_df.itertuples():
            text_id = row.text_id
            text_name = row.text_name
            doc = row.spacy_text
            
            try:        
                # extract the quotes
                quotes = extract_quotes(doc_id=text_id, doc=doc, 
                                        write_tree=create_tree)#, 
                                        #tree_dir=tree_dir)
                
                # extract the included named entities
                speaks, qts = [quote['speaker'] for quote in quotes], [quote['quote'] for quote in quotes]
                speak_ents = self.extract_inc_ent(speaks, doc, inc_ent)
                quote_ents = self.extract_inc_ent(qts, doc, inc_ent)
        
                # add text_id, quote_id and named entities to each quote
                for n, quote in enumerate(quotes):
                    quote['text_id'] = text_id
                    quote['text_name'] = text_name
                    quote['quote_id'] = str(n)
                    quote['speaker_entities'] = list(set(speak_ents[n]))
                    quote['quote_entities'] = list(set(quote_ents[n]))
                    
                # store them in all_quotes
                all_quotes.extend(quotes)
                    
            except:
                # this will provide some information in the case of an error
                self.app_logger.exception("message")
                traceback.print_exception()
                
        # convert the outcome into a pandas dataframe
        self.quotes_df = pd.DataFrame.from_dict(all_quotes)
        
        # convert the string format quote spans in the index columns to a tuple of integers
        for column in self.quotes_df.columns:
            if column.endswith('_index'):
                self.quotes_df[column].replace('','(0,0)', inplace=True)
                self.quotes_df[column] = self.quotes_df[column].apply(eval)
        
        # re-arrange the columns
        new_index = ['text_id', 'text_name', 'quote_id', 'quote', 'quote_index', 'quote_entities', 
                     'speaker', 'speaker_index', 'speaker_entities',
                     'verb', 'verb_index', 'quote_token_count', 'quote_type', 'is_floating_quote']
        self.quotes_df = self.quotes_df.reindex(columns=new_index)
                
        return self.quotes_df
    
    
    def show_entities(self, spacy_doc, selTokens, inc_ent):
        '''
        Add included named entities to displaCy code

        Args:
            spacy_doc: spaCy's processed text for the above list of string
            selTokens: options to display speakers, quotes or named entities
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # empty loist to hold entities code
        ent_code_list = []
        
        # create span code for entities
        for ent in spacy_doc.ents:
            if (ent.start in selTokens) & (ent.label_ in inc_ent):
                span_code = "Span(doc, {}, {}, '{}'),".format(ent.start, 
                                                  ent.end, 
                                                  ent.label_) 
                ent_code_list.append(span_code)
        
        # combine codes for all entities
        ent_code = ''.join(ent_code_list)
        
        return ent_code
    
    
    def show_quotes(self, text_id, show_what, inc_ent):
        '''
        Display speakers, quotes and named entities inside the text using displaCy

        Args:
            text_id: the text_id of the text you wish to display
            show_what: options to display speakers, quotes or named entities
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # formatting options
        TPL_SPAN = '''
        <span style="font-weight: bold; display: inline-block; position: relative; 
        line-height: 55px">
            {text}
            {span_slices}
            {span_starts}
        </span>
        '''
        
        TPL_SPAN_SLICE = '''
        <span style="background: {bg}; top: {top_offset}px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
        </span>
        '''
        
        TPL_SPAN_START = '''
        <span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
            <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
                {label}{kb_link}
            </span>
        </span>
        '''
        
        colors = {'QUOTE': '#66ccff', 'SPEAKER': '#66ff99'}
        options = {'ents': ['QUOTE', 'SPEAKER'], 
                   'colors': colors, 
                   'top_offset': 42,
                   'template': {'span':TPL_SPAN,
                               'slice':TPL_SPAN_SLICE,
                               'start':TPL_SPAN_START},
                   'span_label_offset': 14,
                   'top_offset_step':14}
        
        # get the spaCy text 
        doc = self.text_df[self.text_df['text_id']==text_id]['spacy_text'].to_list()[0]
        
        # create a mapping dataframe between the character index and token index from the spacy text.
        loc2tok_df = pd.DataFrame([(t.idx, t.i) for t in doc], columns = ['loc', 'token'])
    
        # get the quotes and speakers indexes
        locs = {
            'QUOTE': self.quotes_df[self.quotes_df['text_id']==text_id]['quote_index'].tolist(),
            'SPEAKER': set(self.quotes_df[self.quotes_df['text_id']==text_id]['speaker_index'].tolist())
        }
    
        # create the displaCy code to visualise quotes and speakers
        my_code_list = ['doc.spans["sc"] = [', ']']
        
        for key in locs.keys():
            for loc in locs[key]:
                if loc!=(0,0):
                    # Find out all token indices that falls within the given span (variable loc)
                    selTokens = loc2tok_df.loc[(loc[0]<=loc2tok_df['loc']) & (loc2tok_df['loc']<loc[1]), 'token'].tolist()
                    
                    # option to display named entities only
                    if show_what==['NAMED ENTITIES']:
                        ent_code = self.show_entities(doc, selTokens, inc_ent)
                        my_code_list.insert(1,ent_code)
                    
                    # option to display speaker and/or quotes and/or named entities
                    elif key in show_what:
                        if 'NAMED ENTITIES' in show_what:
                            ent_code = self.show_entities(doc, selTokens, inc_ent)
                            my_code_list.insert(1,ent_code)
                        
                        start_token, end_token = selTokens[0], selTokens[-1] 
                        span_code = "Span(doc, {}, {}, '{}'),".format(start_token, end_token+1, key) 
                        my_code_list.insert(1,span_code)
                    
        # combine all codes
        my_code = ''.join(my_code_list)
    
        # execute the code
        exec(my_code)
        
        # display the preview in this notebook
        displacy.render(doc, style='span', options=options, jupyter=True)
        self.html = displacy.render(doc, style='span', options=options, jupyter=False, page=True)
        
    def visualize_entities(self, text_name, which_ent, ent_type, top_n, top_ent, most_ent):
        if top_ent!={}:
            # visualize the top entities            
            bar_colors = {'speaker_entities':'#2eb82e',
                          'quote_entities':'#008ae6'}
            ent_types = {'name': 'entity names',
                         'label': 'entity types'}
            display_height = top_n/2#min(650,top_n*len(max(top_ent.keys(), key=len))/4)
            #print('display_width:',display_width)
            display_width = top_n/1.5
            range_tick = max(1,round(max(top_ent.values())/5))
            #plt.figure(figsize=(max(2.5,display_width), max(10,display_height)))
            
            
            
            plt.figure(figsize=(10, max(5,display_height)))
            plt.barh(list(top_ent.keys()), list(top_ent.values()), color=bar_colors[which_ent])
            for i, v in enumerate(list(top_ent.values())):
                plt.text(v+(len(str(v))*0.05), i, str(v), fontsize=12)
            plt.yticks(fontsize=12)
            plt.xticks(range(0, max(top_ent.values())+range_tick, range_tick), fontsize=12)
            plt.title('Top {} {} entities ({}) in {}'.format(min(top_n,len(top_ent.keys())),which_ent[:-9],ent_types[ent_type],text_name)
                      , fontsize=14)
            plt.show()
        else:
            print('No entities identified in the {}s.'.format(which_ent[:-9]))
        
        
    def top_entities(self, text_id, which_ent, ent_type, top_n=5):
        '''
        Display top n named entities inside the text using displaCy

        Args:
            text_id: the text_id of the text you wish to display
            which_ent: option to display named entities in speakers ('speaker_entities') 
                       or quotes ('quote_entities')
            top_n: the number of entities to display
        '''
        # get the top entities
        if text_id=='all':
            most_ent = self.quotes_df[which_ent].to_list()
            text_name = 'all texts'
        else:
            most_ent = self.quotes_df[self.quotes_df['text_id']==text_id][which_ent].tolist()
            text_name = self.quotes_df[self.quotes_df['text_id']==text_id]['text_name'].to_list()[0]
        
        most_ent = list(filter(None,most_ent))
        most_ent = [ent for most in most_ent for ent in most]
        
        if ent_type=='name':
            most_ent = Counter([ent_name for ent_name, ent_label in most_ent])
        if ent_type=='label':
            most_ent = Counter([ent_label for ent_name, ent_label in most_ent])
        
        #top_ent = dict(most_ent.most_common(top_n)
        top_ent = dict(sorted(most_ent.items(), key=lambda x: x[1], reverse=False)[-top_n:])
        self.visualize_entities(text_name, which_ent, ent_type, top_n, top_ent, most_ent)


    def analyse_quotes(self, inc_ent):
        '''
        Interactive tool to display and analyse speakers, quotes and named entities inside the text

        Args:
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # widget for selecting text_id
        enter_text = widgets.HTML(
            value='<b>Select which text to preview:</b>',
            placeholder='',
            description=''
            )
        
        text_options = self.text_df.text_name.to_list() # get the list of text_names
        text = widgets.Combobox(
            placeholder='Choose text to analyse...',
            options=text_options,
            description='',
            ensure_option=True,
            disabled=False,
            layout = widgets.Layout(width='195px')
        )
        
        # widgets to select what to preview, i.e., speaker and/or quote and/or named entities
        entity_options = widgets.HTML(
            value="<b>Select which entity to show:</b>",
            placeholder='',
            description='',
            )
        
        speaker_box = widgets.Checkbox(
            value=False,
            description='Speaker',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        quote_box = widgets.Checkbox(
            value=False,
            description='Quote',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        ne_box = widgets.Checkbox(
            value=False,
            description='Named Entities',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        # widget to show the preview
        preview_button = widgets.Button(description='Preview', 
                                        layout=Layout(margin='10px 0px 0px 10px'),
                                        style=dict(font_style='italic',
                                                   font_weight='bold'))
        preview_out = widgets.Output()
        
        def on_preview_button_clicked(_):
            with top_out:
                # remove bar plot if previewing new text_id
                if text.value!=self.current_text:
                    clear_output()
                    self.current_text = text.value
            
            with save_out:
                clear_output()
            
            with preview_out:                                                                                   
                # what happens when we click the preview_button
                clear_output()
                text_name = text.value
                text_id = self.quotes_df[self.quotes_df['text_name']==text_name]['text_id'].to_list()[0]
                show_what = []
                if speaker_box.value:
                    show_what.append('SPEAKER')
                if quote_box.value:
                    show_what.append('QUOTE')
                if ne_box.value:
                    show_what.append('NAMED ENTITIES')
                if show_what==[]:
                    print('Please select the entities to display!')
                else:
                    try:
                        self.show_quotes(text_id, show_what, inc_ent)
                    except:
                        print('Please enter the correct text_id!')
        
        # link the preview_button with the function
        preview_button.on_click(on_preview_button_clicked)
        
        # widget to save the preview
        save_button = widgets.Button(description='Save Preview', 
                                     layout=Layout(margin='10px 0px 0px 10px'),
                                     style=dict(font_style='italic',
                                                font_weight='bold'))
        
        save_out = widgets.Output()
        
        def on_save_button_clicked(_):
            with save_out:
                try:
                    # create an output folder if not yet available
                    os.makedirs('output', exist_ok=True)
                    out_dir='./output/'
                    text_name = text.value
                    text_id = self.quotes_df[self.quotes_df['text_name']==text_name]['text_id'].to_list()[0]
                    
                    # save the preview as an html file
                    file = open(out_dir+str(text_name)+'.html', 'w')
                    file.write(self.html)
                    file.close()
                    clear_output()
                    print('Preview saved!')
                except:
                    print('You need to generate a preview before you can save it!')
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # widget to show top 5 entities
        top_button = widgets.Button(description='Top 5 Entities', 
                                     layout=Layout(margin='10px 0px 0px 10px'),
                                     style=dict(font_style='italic',
                                                font_weight='bold'))
        top_out = widgets.Output()
        
        # displaying buttons and their outputs
        vbox1 = widgets.VBox([enter_text, text, entity_options, speaker_box, quote_box, ne_box,
                              preview_button, save_button])
        
        hbox = widgets.HBox([vbox1])
        vbox = widgets.VBox([hbox, save_out, preview_out])
        
        return vbox
    
    def analyse_entities(self, inc_ent):
        '''
        Interactive tool to display and analyse named entities inside the text

        Args:
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # widget for selecting text_id
        enter_text = widgets.HTML(
            value='<b>Select which text to preview:</b>',
            placeholder='',
            description=''
            )
        
        text_options = self.text_df.text_name.to_list() # get the list of text_names
        text_options.insert(0, 'all')
        text = widgets.Combobox(
            placeholder='all',
            options=text_options,
            description='',
            ensure_option=True,
            disabled=False,
            layout = widgets.Layout(width='195px')
        )
        
        # widgets to select what to preview, i.e., speaker and/or quote and/or named entities
        entity_options = widgets.HTML(
            value="<b>Select which entity to show:</b>",
            placeholder='',
            description='',
            )
        
        speaker_box = widgets.Checkbox(
            value=False,
            description='Speaker',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        quote_box = widgets.Checkbox(
            value=False,
            description='Quote',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        label_options = widgets.HTML(
            value="<b>Select what you want to display:</b>",
            placeholder='',
            description='',
            )
        name_box = widgets.Checkbox(
            value=False,
            description='Entity names (e.g., John Doe, Sydney, etc.)',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        entity_box = widgets.Checkbox(
            value=False,
            description='Entity types (e.g., PERSON, ORG, etc.)',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        enter_n = widgets.HTML(
            value='<b>How many entities to display:</b>',
            placeholder='',
            description=''
            )
        
        top_n_option = widgets.BoundedIntText(
            value=5,
            min=0,
            #max=1000,
            step=5,
            description='',
            disabled=False,
            layout = widgets.Layout(width='100px')
        )
                
        # widget to show top 5 entities
        top_button = widgets.Button(description='Show Top Entities', 
                                     layout=Layout(margin='10px 0px 0px 10px'),
                                     style=dict(font_style='italic',
                                                font_weight='bold'))
        top_out = widgets.Output()
        
        def on_top_button_clicked(_):
            with top_out:
                # what happens when we click the top_button
                clear_output()
                text_name = text.value
                top_n = top_n_option.value
                if text_name=='all':
                    text_id='all'
                else:
                    text_id = self.quotes_df[self.quotes_df['text_name']==text_name]['text_id'].to_list()[0]
                which_ents=[]; ent_types=[]
                try:
                    if quote_box.value:
                        which_ents.append('quote_entities')
                    if speaker_box.value:
                        which_ents.append('speaker_entities')
                    if name_box.value:
                        ent_types.append('name')
                    if entity_box.value:
                        ent_types.append('label')
                    for ent_type in ent_types:
                        for which_ent in which_ents:
                            self.top_entities(text_id, which_ent, ent_type, top_n)
                except:
                    print('Please select which entities to display and whether to display actual or entity names!')
        
        # link the top_button with the function
        top_button.on_click(on_top_button_clicked)
        
        # displaying buttons and their outputs
        vbox1 = widgets.VBox([enter_text, text, 
                              entity_options, speaker_box, quote_box, 
                              label_options, name_box, entity_box,
                              enter_n, top_n_option, 
                              top_button])
        
        vbox = widgets.VBox([vbox1, top_out])
        
        return vbox
    