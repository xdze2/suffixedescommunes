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
from collections import Counter
import random

# # Carte des communes françaises ayant un même suffixe
# Répartition géographique des suffixes des noms des communes françaises
#
# ## 1. Chargement des données
#

filename = "data/code-insee-postaux-geoflar.csv"
# https://public.opendatasoft.com/explore/dataset/code-insee-postaux-geoflar/export/

with open(filename, 'r') as csvfile:  # python 3: 'r',newline=""
    dialect = csv.Sniffer().sniff(csvfile.read(2024), delimiters=";,")
    csvfile.seek(0)
    reader = csv.reader(csvfile, dialect)
    header = reader.__next__()
    data = {key:col for key, col in zip(header, zip(*[line for line in reader]))}

print(len(list(data.values())[0]), 'rows')
print('> Columns:')
print(', '.join( data.keys() ))


# +
# Extract data
def remove_parenthesis(n):
    # remove parenthesis, example 'Castillon (Canton de Lembeye)'
    return n.split('(')[0].strip()

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
    
print(len(name_xy))

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

# +
select = [n for n in names if n.endswith('lieu')]
print(len(select))

print( select[:100] )
# -

sorted(names, key=len, reverse=True)[:10]

sorted(name_xy.items(), key=lambda x:len(x[1][0]), reverse=False)[:10]

# ## 2. Recherche des suffixes
#
# branching entropy:  https://www.aclweb.org/anthology/P12-2075.pdf

# +
# pre-compute counters of n-grams (suffix)
len_max = 15
count_k = [Counter( (n[-k:] for n in names if len(n)>=k) )
           for k in range(1, len_max)]
alphabet = sorted( count_k[0].keys() )

# print summary
for k, counter in enumerate(count_k):
    print(f'{k+1:>3}',
          f'{len(counter):>7}',
          #f'{entropy(counter):>7.2f}', 
          ', '.join(u[0] for u in counter.most_common()[:4]))


# +
def get_count(suffix):
    n = len(suffix) - 1
    return count_k[n][suffix]

left_entropy_store = {}
def left_entropy(suffix):
    """Left Branching Entropy"""
    if suffix in left_entropy_store:
        return left_entropy_store[suffix]
    else:
        n_suffix = get_count(suffix)
            
        if n_suffix < 1:
            left_entropy = 0
        else:
            # probabilty p(letter | suffix)
            p_left = (get_count(letter+suffix)/n_suffix for letter in alphabet)
            p_left = np.array([p for p in p_left if p > 0])
            left_entropy = -np.sum( p_left*np.log(p_left) )
            
            left_entropy_store[suffix] = left_entropy
        return left_entropy

def VBE(suffix): 
    """Variation of Branching Entropy"""
    return left_entropy(suffix) - left_entropy(suffix[1:])

def compute_VBE_avg(k):
    """mean of the Variation of Branching Entropy for k-grams"""
    total = sum(count_k[k].values())
    mu = sum(VBE(suffix)*count for suffix, count in count_k[k].items() if count>100)
    return mu / total

# pre-compute average VBE_k
VBE_avg = [compute_VBE_avg(k) for k in range(1, len(count_k))]
print(VBE_avg)


# -

def normed_VBE(suffix):
    return VBE(suffix) - VBE_avg[len(suffix)]


# +
VBE_k = [(suffix, count, normed_VBE(suffix)) for k in range(1, 12)
         for suffix, count in count_k[k].most_common()[:70] ]

VBE_k = [vbe for vbe in VBE_k if vbe[2] > 0 and vbe[1]>60]
VBE_k = sorted(VBE_k, key=lambda x:x[2], reverse=True)
# -

VBE_k

# ## 3. Dessin des cartes

# !mkdir images/vbe

xy_choices = [xy for n, xy in name_xy.values()]

i = 1
a, b = min(xy_choices, key=lambda x:x[i])[i], max(xy_choices, key=lambda x:x[i])[i]

L = 11400

uniqueletter = [(n, c, 0) for n, c in count_k[0].items()
                if c > 60]

for suffix, c, vbe in VBE_k+uniqueletter:
    if c < 60 or not suffix:
        continue
    print(suffix+' '*10, end='\r')
    fig = plt.figure(figsize=(7, 7));
    ax = fig.add_axes([0., 0., 1., 1 ])
    margin = 300
    ax.axis([1009.5-margin, 12409.5+margin, 60130.5-margin, 71530.5+margin]);
    ax.axis('off')
    xy_choices = [xy for n, xy in name_xy.values()]
    ax.plot(*zip(*xy_choices), ',',
             color= 'black',
             alpha= 0.25,
             markersize=1,
             label=f'-{suffix}');

    xy_choices = [xy for n, xy in name_xy.values() if n.endswith(suffix)]
    ax.plot(*zip(*xy_choices),
             color='firebrick',
             alpha=.8,
             markersize=1, marker='s', linestyle='',
             label=f'-{suffix}');

    count = len(xy_choices)
    #plt.axis('equal'); #plt.title(f'{c} communes avec comme suffix -{suffix}');
    #text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
    #               'fontweight': 'bold'
    ax.text(4300, 70400, f'*{suffix}', fontsize=16,
             fontweight='bold', family='monospace',
             color='firebrick', ha='right')
    ax.text(4300, 70050, f'{count} communes', fontsize=11,
             fontweight='normal', family='monospace',
             color='firebrick', ha='right', alpha=.9)
    #plt.tight_layout();
    plt.savefig(f"images/vbe/{''.join(suffix[::-1])}_{count}.png")
    
    #break
    plt.close();

suffix = 'abc'

''.join(suffix[::-1])

# ## map all k-endgram 

k = 1
for suffix, c in count_k[k-1].items():
    if c < 200 or not suffix:
        continue
    print(suffix, end='\r')
    plt.figure(figsize=(8, 8));

    xy_choices = [xy for n, xy in name_xy.values() if random.random()>.0]
    plt.plot(*zip(*xy_choices), ',',
             color='black',
             alpha=.25,
             markersize=1,
             label=f'-{suffix}');

    xy_choices = [xy for n, xy in name_xy.values() if n.endswith(suffix)]
    plt.plot(*zip(*xy_choices),
             color='firebrick',
             alpha=.8,
             markersize=1, marker='s', linestyle='',
             label=f'-{suffix}');
    

    plt.axis('equal'); plt.title(f'suffix -{suffix}  {c} villes');
    plt.axis(False);
    plt.tight_layout();
    plt.savefig(f'images/{len(suffix)}/france_{suffix}.png')
    
    plt.close();

# ## test Map

import matplotlib.colors as mcolors

colors = list(mcolors.XKCD_COLORS.keys())

colors = list(mcolors.TABLEAU_COLORS.keys())

print( colors )


def pick_a_color():
    return random.choice(colors)


def plot_points(suffix, color):
    xy_choices = [xy for n, xy in name_xy.values() if n.endswith(suffix)]
    plt.plot(*zip(*xy_choices), '.',
             color=color,
             alpha=.8,
             markersize=3,
             label=f'-{suffix}');


# +
plt.figure(figsize=(8, 8));

plot_points('o', pick_a_color())
plot_points('m', pick_a_color())
plot_points('ville', pick_a_color())
plot_points('ieu', 'xkcd:blue')
plot_points('ac', pick_a_color())
plot_points('y', pick_a_color())

plot_points('ix', 'yellow')

plt.axis('equal'); plt.legend();
# -

selection = ['ville', 'heim', 'court', 'a', 'o', 'ac', 'ieu',
             '-sur-mer', 'ing',  'ans', 'i', 'loire']
#selection = ['loire', 'rhone', 'a', 'o']

# +
fig = plt.figure(figsize=(7, 7));
ax = fig.add_axes([0., 0., 1., 1 ])

margin = 300
ax.axis([1009.5-margin, 12409.5+margin, 60130.5-margin, 71530.5+margin]);
ax.axis('off')
xy_choices = [xy for n, xy in name_xy.values()]
ax.plot(*zip(*xy_choices), ',',
         color= 'black',
         alpha= 0.15,
         markersize=1,
         label=f'-{suffix}');
    
for k, suffix in enumerate(selection):

    #print(suffix+' '*10, end='\r')
    color = colors[k % len(colors)]

    xy_choices = [xy for n, xy in name_xy.values() if n.endswith(suffix)]
    ax.plot(*zip(*xy_choices),
             color=color,
             alpha=.9,
             markersize=1, marker='s', linestyle='',
             label=f'-{suffix}');

    #count = len(xy_choices)
    #plt.axis('equal'); #plt.title(f'{c} communes avec comme suffix -{suffix}');
    #text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
    #               'fontweight': 'bold'
    ax.text(2300, 66800 - k*350, f'*{suffix}', fontsize=16,
             fontweight='bold', family='monospace',
             color=color, ha='right')
    #ax.text(4300, 70050, f'{count} communes', fontsize=11,
    #         fontweight='normal', family='monospace',
    #         color='firebrick', ha='right', alpha=.9)
    #plt.tight_layout();
    #plt.savefig(f'images/vbe/{suffix}.png')
    
    #break
    #plt.close();


# -

# ## Draft

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
    lastcounter = Counter( n[len(n)-len(ends)-1:len(n)-len(ends)] for n in names if n.endswith(ends) )
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
ends = '-sur-loire'
last = count_from_ends(ends)
print("suffix:", f'-{ends}')
nbr = sum(last.values())
print("   nbr:", nbr)
print("     e=", entropy(last))
d = [(letter, c, entropy(count_from_ends(letter+ends))) for letter, c in last.items()]
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


