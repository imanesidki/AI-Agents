from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

invoice_text = """Facture 5 3 silà

orange
C BIS.COM
Numéro de Facture : F-0124-1268258 ANG.ROUDANI N°
N° Compte client : 12048731 RUE ABOU SALT EL ANDALOUSSI
Date d'édition : 02/01/2024 20000 CASABLANCA

Vos coordonnées

cbiscom


ANG.ROUDANI N° votre facture du 1 Janvier 2024
RUE ABOU SALT EL ANDALOUSSI Montant total à payer 649,00 DH
ICE : 001581583000054 Date limite de paiement le 16/01/2024 #10 Jai st
Nous contacter : Ë "

Pour votre sécurité en cette période, nous vous
Par courrier e pe a es en li r'
Orange Maroc recommandons de payer vos factures en ligne. Votre

Lotissement la colline 11 écurité est notre priorité
Immeuble les Quatre Temps

Votre service clients

178 depuis un mobile Résumé de la facture Montant en DH
Orange (appel gratuit)
(+212) 05 20 178 178 m Vos forfaits/abonnements et options 540,83
+coût appel selon opérateur
7 j/7 de 07h à 00h Montant HT cs 540,83
relationentreprise@orange.ma TVA (20%) eue 108,17
www.entreprise.orange ma
Total TTC 649,00
Solde en votre faveur * 0,00
Montant total à payer o3lal cast ll all 649,00

* Palements effectués par le client

A noter Vos avantages
Découvrez le nouveau Pack Pro d'Orange en allant sur le site : Bénéficiez d'appels en illimité entre les collaborateurs de votre
hitps://entreprise.orange.ma/ entreprise
Rechargez votre ligne ou celles de vos collaborateurs via les Pass
Orangr

je
Récupérez votre TVA en tant que client Entreprise

MEDI TELECOM, SA au Capital de 2.373.168.700 DH Page 1.4 5
Siège social: Lotissement la Coline Il, Immeuble les Quatre Temps, Sidi Maërouf, Casablanca, 20270, Maroc
RC 97815 - Patente 37 998 011 — IF 108 6826 - CNSS 6018924 - ICE 001524628000007

snge ma
"""

def process_invoice(invoice_text):
    model = Ollama(model = "llama3.1")

    Extractor = Agent(
        role = "Invoice Extractor",
        goal = "Extract invoice details from text",
        backstory= " You are an invoice extraction agent. You are required to extract the invoice details from the given invoice formatted text.",
        verbose = True,
        allow_delegation= False,
        llm= model
    )

    Responder = Agent(
        role = "Invoice Responder",
        goal = "Respond by a json containing the invoice details extracted, respond with only json without any text.",
        backstory="You are an API endpoint whose only job is to respond with a json containing the invoice details extracted that will be provided to you by the 'Invoice Extractor' agent.",
        verbose = True,
        allow_delegation= False,
        llm= model
    )

    extract_invoice = Task(
        description=f"Extract invoice details from the given invoice text: '{invoice_text}'.",
        agent = Extractor,
        expected_output= "Invoice details extracted from the given invoice text."
    )

    respond_invoice = Task(
        description=f"Respond to the '{invoice_text}' with a json containing the invoice details extracted and provided by the 'Extractor' agent. The response should be only the json and no other text.",
        agent = Responder,
        expected_output= """Json containing the invoice details extracted as bellow: 
    {
    'invoiceClient': { // info about the receiver of the invoice
    'name': 'string', // name of the receiver e.g. C BIS.COM or INFLEXIT
    'ICE':'number',
    'email': 'string',
    'invoiceAddress': 'string',
    'shippingAddress': 'string'
    },
    'invoiceAt': "YYYY-MM-DD", // invoice date
    'invoiceDueAt': "YYYY-MM-DD", // invoice due date
    'amount': 'number', // total price of the entire invoice including taxes
    'invoiceItems': [
    {
        'unitPrice': 'number',
        'quantity': 'number',
        'tva': 'number',
        'discount': 'number',
        'netAmount': 'number',   
        'amount': 'number',
        'name': 'string'
    }
    ]
    }
    in case you didn't find a specific value, it should be null for strings and 0 for numbers"""
    )

    crew = Crew(
        agents=[Extractor, Responder],
        tasks=[extract_invoice, respond_invoice],
        verbose= 0,
        process= Process.sequential # Tasks are executed sequentially
    )

    output = crew.kickoff()
    return output


output = process_invoice(invoice_text)
print(output)