# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import csv
import matplotlib.pylab as plt
from collections import Counter, defaultdict
import random

# # Carte des communes françaises ayant un même suffixe
# Répartition géographique des suffixes des noms des communes françaises
#
# ## 1. Chargement des données
#

filename = "data/code-insee-postaux-geoflar.csv"
# source:
# https://public.opendatasoft.com/explore/dataset/code-insee-postaux-geoflar/export/

# +
with open(filename, 'r') as csvfile:
    # detect delimiter
    dialect = csv.Sniffer().sniff(csvfile.read(2024), delimiters=";,")
    csvfile.seek(0)
    reader = csv.reader(csvfile, dialect)
    header = reader.__next__()
    data = {key:col for key, col in zip(header, zip(*[line for line in reader]))}
    
print(len(list(data.values())[0]), 'rows')
# -

print('> Columns:')
print(', '.join( data.keys() ))


# +
# -- Extract data --
def remove_parenthesis(n):
    # remove parenthesis, example 'Castillon (Canton de Lembeye)'
    return n.split('(')[0].strip()

# note: paris-2e-arrondissement, marseille--5e--arrondissement, ...

name_xy = {}
for code, name, x, y in zip(data['CODE INSEE'], data['Nom Commune'],
                            data['X Centroid'], data['Y Centroid']):
    if not name or len(name) <= 1:
        continue
    try:
        x = float(x)
        y = float(y)
        name_xy[code] = (remove_parenthesis(name.lower()), (x, y))
    except ValueError:
        pass
    
print('nbr of names:', len(name_xy))

names = [n for n, xy in name_xy.values()]
# -

# ## simple stats

mostcommonname = Counter(names)
mostcommonname.most_common()[:10]

nouns = ['saint', 'sainte', 'ville', 'bourg', 'mont',
         'pierre', 'puy', 'port', 'fort', 'sur', 'bel', 'beau', 'lieu']
for part in nouns:
    c = len([n for n in names if part in n])
    print(part, f'\t{c:>5}')

# Noms les plus long :
print('\n'.join( sorted(names, key=len, reverse=True)[:10]))

# et les plus courts :
sorted(name_xy.items(), key=lambda x:len(x[1][0]), reverse=False)[:10]

# ## 2. Recherche automatique des suffixes
#
# using branching entropy:  https://www.aclweb.org/anthology/P12-2075.pdf  
# see https://pypi.org/project/eleve/

# +
# -- Pre-compute counts for each n-grams (suffix) --

suffix_max_size = 15  #  <-- max length of searched n-grams

suffix_counter = Counter({'':len(names)})
for k in range(1, suffix_max_size):
    suffix_counter.update( (n[-k:] for n in names if len(n)>=k ) )
# -

alphabet = Counter( (letter for n in names for letter in n) )

# +
left_entropy_store = {}
def left_entropy(suffix):
    """Left Branching Entropy
    
       considering the probability distribution p( letter+suffix | suffix )
       
       p( letter+suffix | suffix ) = #(letter+suffix)/#(suffix)
    """
    if suffix in left_entropy_store:
        left_entropy = left_entropy_store[suffix]
    else:
        n_suffix = suffix_counter[suffix]
            
        if n_suffix < 20:  # <-- min number of counts to do stats
            left_entropy = np.NaN
        else:
            # probabilty p(letter | suffix)
            p_left = (suffix_counter[letter+suffix]/n_suffix
                      for letter in alphabet.keys())
            p_left = np.array([p for p in p_left if p > 0])
            left_entropy = -np.sum( p_left*np.log(p_left) )
        
        left_entropy_store[suffix] = left_entropy
            
    return left_entropy

def VBE(suffix): 
    """Variation of Branching Entropy"""
    return left_entropy(suffix) - left_entropy(suffix[1:])

def compute_VBE_avg_notused(k):
    """mean of the Variation of Branching Entropy for k-grams"""
    total = sum(count_k[k].values())
    mu = sum(VBE(suffix)*count for suffix, count in count_k[k].items())# if count>100)
    return mu / total
# -



# +
# -- Compute average VBE for k-grams, for every k -- 
weighted_sum = defaultdict(int)
weighted_count = defaultdict(int)
for suffix, count in suffix_counter.items():
    vbe = VBE(suffix)
    if not np.isnan(vbe):
        weighted_count[len(suffix)] += count
        weighted_sum[len(suffix)] += vbe*count

avg_vbe = {k:vbe/weighted_count[k] for k, vbe in weighted_sum.items()}

def normed_VBE(suffix):
    return VBE(suffix) - avg_vbe[len(suffix)]


# -

normed_VBE('ard')

# +
# Compute normed VBE for each suffix
VBE_k = [(suffix, count, normed_VBE(suffix))
         for suffix, count in suffix_counter.items()
         if count > 50]

# Sort and filter
VBE_k = [vbe for vbe in VBE_k if vbe[2] > 0]
VBE_k = sorted(VBE_k, key=lambda x:x[2], reverse=True)
# -

VBE_k[:150]

# +
## Test Graph
# -

import altair as alt
import pandas as pd

# +
suffix, count, vbe = zip(*VBE_k)

chartdata = pd.DataFrame({'suffix': suffix,
                          'count':count,
                          'vbe':vbe})
chart = alt.Chart(chartdata)

alt.Chart(chartdata).mark_point().encode(
    alt.X('count', scale={'type':'log', 'base':10}),
    alt.Y('vbe'),
    alt.Tooltip('suffix')
)
# -

# ## 3. Dessin des cartes

# !mkdir one_by_one

uniqueletter = [(a, suffix_counter[a], 0) for a in alphabet
                if suffix_counter[a] > 60]

for suffix, c, vbe in VBE_k+uniqueletter:
    # filter
    if c < 60 or not suffix or vbe < 0.21:
        continue
    
    print(suffix+' '*10, end='\r')
    
    fig = plt.figure(figsize=(7, 7));
    ax = fig.add_axes([0., 0., 1., 1 ])
    margin = 300
    ax.axis([1009.5-margin, 12409.5+margin, 60130.5-margin, 71530.5+margin]);
    ax.axis('off')
    # background
    xy_choices = [xy for n, xy in name_xy.values()]
    ax.plot(*zip(*xy_choices), ',',
             color= 'black',
             alpha= 0.25,
             markersize=1,
             label=f'-{suffix}');
    # points
    xy_choices = [xy for n, xy in name_xy.values() if n.endswith(suffix)]
    ax.plot(*zip(*xy_choices),
             color='firebrick',
             alpha=.8,
             markersize=1, marker='s', linestyle='',
             label=f'-{suffix}');

    # legend
    count = len(xy_choices)
    ax.text(4300, 70400, f'*{suffix}', fontsize=16,
             fontweight='bold', family='monospace',
             color='firebrick', ha='right')
    ax.text(4300, 70050, f'{count} communes', fontsize=11,
             fontweight='normal', family='monospace',
             color='firebrick', ha='right', alpha=.9)
    #plt.tight_layout();
    plt.savefig(f"one_by_one/{''.join(suffix[::-1])}_{count}.png")
    
    #break
    plt.close();

from glob import glob

filepath = sorted( glob('one_by_one/*.png') )


# +
def reverse_suffix(path):
    suffix = path.split('/')[-1].split('_')[0]
    return suffix[::-1]

path = 'one_by_one/rem-rus-_100.png'
reverse_suffix(path)

# +
# generate .md file
text = '\n'.join([f'### *{reverse_suffix(path)}  \n ![{reverse_suffix(path)}]({path})'
                  for path in filepath])

with open("almost_all.md", 'w') as file: 
    file.write(text) 
# -

# ## Mutliple Map

import matplotlib.colors as mcolors

colors = list(mcolors.XKCD_COLORS.keys())

colors = list(mcolors.TABLEAU_COLORS.keys())

print( colors )

selection = ['ville', 'heim', 'court', 'a', 'o', 'ac', 'ieu',
             '-sur-mer', 'ing',  'ans', 'i', 'loire']
#selection = ['loire', 'rhone', 'a', 'o']

def plot_points(suffix, color, xy, alpha=0.9):
    
    xy_choices = [xy for n, xy in name_xy.values() if n.endswith(suffix)]
    ax.plot(*zip(*xy_choices),
             color=color,
             alpha=alpha,
             markersize=1, marker='s', linestyle='',
             label=f'-{suffix}');

    ax.text(*xy, f'*{suffix}', fontsize=16,
             fontweight='bold', family='monospace',
             color=color, ha='right', alpha=alpha)


# +

fig = plt.figure(figsize=(7, 7));
ax = fig.add_axes([0., 0., 1., 1 ])

margin = 300
ax.axis([1009.5-3*margin, 12409.5+margin, 60130.5-margin, 71530.5+margin]);
ax.axis('off')

# background
xy_choices = [xy for n, xy in name_xy.values()]
ax.plot(*zip(*xy_choices), ',',
         color= 'black',
         alpha= 0.15,
         markersize=1,
         label=f'-{suffix}');
    
plot_points('y', 'tab:gray', (7200, 71300), alpha=.7)
plot_points('ville', 'tab:purple',  (5400, 71200), alpha=.5)

plot_points('a', 'tab:red', (10400, 66200))
plot_points('-sur-mer', 'tab:blue', (5400, 70200))
plot_points('ing', 'tab:orange', (10400, 69500))
plot_points('ec', 'tab:cyan', (1700, 69000))
plot_points('ac', 'tab:brown', (2700, 64000))
plot_points('willer', 'tab:red', (12100, 69000))
plot_points('heim', 'tab:pink', (12000, 67800))
plot_points('an', 'tab:pink', (2700, 63400))
plot_points('ans', 'tab:green', (10900, 66700))
plot_points('os', 'tab:green', (2700, 62800))

plot_points('ieu', 'tab:red', (10800, 65200))

plot_points('at', 'tab:blue', (11400, 65800))
plot_points('court', 'tab:olive', (9300, 70300))

plot_points('-sur-seine', 'tab:red', (5400, 70700))
plot_points('-sur-loire', 'tab:red', (2800, 66400))

plot_points('o', 'tab:blue', (11400, 60800))
plot_points('i', 'tab:green', (11400, 61200))


#plt.tight_layout();
plt.savefig('not_all_suffix.png')

# -

# ## Test VBE 

def entropy(counter):
    count = np.array(list( counter.values() ))
    total = np.sum(count)
    p = count/total
    entropy = -np.sum( p * np.log( p ) )
    return entropy


# +
def iter_kgram(n, k):
    return (n[i:i+k] for i in range(len(n)-k+1))

list( iter_kgram('abcde', 2) )
# -

k = 3
k_grams = Counter( ngram for n in names for ngram in iter_kgram(n, k) )
print("entropy", entropy(k_grams))
print("    max", np.log( 26 )*k)
print("  ratio", entropy(k_grams)/np.log( len(k_grams) )*100)


def count_from_ends(ends):
    lastcounter = Counter( n[len(n)-len(ends)-1:len(n)-len(ends)]
                          for n in names if n.endswith(ends) )
    #print(len(lastcounter))
    return lastcounter


# +
def bar_chart(ratio, full_length=20):
    bar_length = ratio * full_length
    n_bulk = int(np.floor(bar_length))
    partials = ['', '.', ':', ':.', '#']
    remaining = bar_length - n_bulk
    idx = int(np.round(remaining*4))
    return '|' + '#'*n_bulk + partials[idx]

# test
ratio = 2.51/2/100
print(bar_chart(ratio))

# +
ends = 'lle'
last = count_from_ends(ends)
print("suffix:", f'-{ends}')
nbr = sum(last.values())
print("   nbr:", nbr)
print("     e=", entropy(last))
d = [(letter, c, entropy(count_from_ends(letter+ends)))
     for letter, c in last.items()]
d = sorted(d, key=lambda x:x[2], reverse=False)

count_max = max(u[1] for u in d)
for letter, count, e in d:
    print(letter+'-'+ends,
          f'{e: .3f}',
          f'{count:>5}',#f'{count/nbr*100:>7.1f}',
          bar_chart(count/count_max))
# -

normed_VBE('-sur-loire')

# +
# final ngram

# +
k = 4

tuplescount = Counter( n[-k:] for n in names )
print(len(tuplescount), 'unique' , sum(tuplescount.values()))


d = [(ngram, c, entropy(count_from_ends(ngram)))
     for ngram, c in tuplescount.most_common()[:50]]
d = sorted(d, key=lambda x:x[1], reverse=True)

for ngram, count, e in d[:30]:
    print(ngram, f'{e: .3f}', f'({count:d})')
    
# -

plt.figure(figsize=(10, 10))
for ngram, count, e in d:
    plt.plot(np.log(count), e, '.')
    plt.annotate(ngram, (np.log(count), e))#, \*args, \*\*kw


def get_ngram_e(k):
    tuplescount = Counter( n[-k:] for n in names )
    nbr_ngram_unique = len(tuplescount)
    print(nbr_ngram_unique, 'unique' , sum(tuplescount.values()))

    e_left = entropy(count_from_ends)
    d = [(ngram, c/nbr_ngram_unique, entropy(count_from_ends(ngram)))
         for ngram, c in tuplescount.most_common()[:100]]
    d = sorted(d, key=lambda x:x[1], reverse=True)
    return d


# +
plt.figure(figsize=(10, 10))

for ngram, count, e in get_ngram_e(3):
    plt.plot(np.log(count), e, '.')
    plt.annotate(ngram, (np.log(count), e))#, \*args, \*\*kw
    
for ngram, count, e in get_ngram_e(4):
    plt.plot(np.log(count), e, '.')
    plt.annotate(ngram, (np.log(count), e))#, \*args, \*\*kw
    
for ngram, count, e in get_ngram_e(5):
    plt.plot(np.log(count), e, '.')
    plt.annotate(ngram, (np.log(count), e))#, \*args, \*\*kw
    
# -


