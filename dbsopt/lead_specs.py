#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:33:20 2022

@author: skyjones
"""

def obj_dict(name,material,is_contact,color):
    
    d = {'name':name, 'material':material, 'is_contact':is_contact, 'color':color}
    return d


def rod_body(name,
             material='Tecothane 75D (Polyurethane)',
             is_contact=False,
             color=(90,90,90,1)): # gray
    return obj_dict(name,material,is_contact,color)


def insulator(name,
             material='Tecothane 75D (Polyurethane)',
             is_contact=False,
             color=(192,87,255,1)): # purple
    return obj_dict(name,material,is_contact,color)


def contact(name,
             material='Platinium - Iridium',
             is_contact=True,
             color=(225,225,225,1)): # off-white
    return obj_dict(name,material,is_contact,color)


MEDTRONIC_B33005 = [
    
    rod_body(name='tip'), # object 0
    rod_body(name='bottom_core'), # object 1
    insulator(name='vert_sep_1'), # object 2
    insulator(name='hor_sep_1'), # object 3
    insulator(name='hor_sep_2'), # object 4
    insulator(name='hor_sep_3'), # object 5
    insulator(name='vert_sep_2'), # object 6
    insulator(name='hor_sep_4'), # object 7
    insulator(name='hor_sep_5'), # object 8
    insulator(name='hor_sep_6'), # object 9
    insulator(name='vert_sep_3'), # object 10
    rod_body(name='midshaft_bottom'), # object 11
    rod_body(name='midshaft_bigtop_1'), # object 12
    rod_body(name='midshaft_bigtop_2'), # object 13
    rod_body(name='midshaft_topcore'), # object 14
    rod_body(name='top'), # object 15
    contact(name='contact_1'), # object 16
    contact(name='contact_3'), # object 17
    contact(name='contact_4'), # object 18
    contact(name='contact_2'), # object 19
    contact(name='contact_6'), # object 20
    contact(name='contact_7'), # object 21
    contact(name='contact_5'), # object 22
    contact(name='contact_8'), # object 23
    rod_body(name='midshaft_smalltop_1', color=(255,255,87,1)), # object 24, yellow
    rod_body(name='midshaft_smalltop_2', color=(255,255,87,1)), # object 25, yellow
    
    ]
    
    
    
    