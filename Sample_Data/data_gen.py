from reportlab.lib.pagesizes import A4

from reportlab.pdfgen import canvas

from faker import Faker

import random

import os


fake = Faker("en_IN")


# Create output folder

output_folder = "customer_feedback_forms"

os.makedirs(output_folder, exist_ok=True)


questions = [

    "Employee greeted you politely",

    "Employee listened carefully to your concerns",

    "Employee explained banking services clearly",

    "Employee behaved professionally",

    "Employee resolved your issue efficiently",

    "Overall satisfaction with employee behavior"

]


branches = [

    "Kolkata Main Branch",

    "Salt Lake Branch",

    "Patna Branch",

    "Howrah Branch",

    "Bhubaneswar Branch"

]


for i in range(1, 101):


    customer_name = fake.name()

    account_number = str(random.randint(1000000000, 9999999999))

    contact_number = fake.phone_number()

    branch = random.choice(branches)


    pdf_file = os.path.join(

        output_folder,

        f"Customer_Feedback_{i:03d}.pdf"

    )


    c = canvas.Canvas(pdf_file, pagesize=A4)

    width, height = A4


    # Title

    c.setFont("Helvetica-Bold", 18)

    c.drawCentredString(

        width / 2,

        height - 50,

        "Customer Feedback Form"

    )


    # Bank Details

    c.setFont("Helvetica", 11)

    c.drawString(50, height - 90,

                 "Bank Name: ABC National Bank")

    c.drawString(50, height - 115,

                 f"Branch Name: {branch}")

    c.drawString(50, height - 140,

                 f"Date: {fake.date_between(start_date='-1y', end_date='today')}")


    # Customer Information

    c.setFont("Helvetica-Bold", 14)

    c.drawString(50, height - 180,

                 "Customer Information")


    c.setFont("Helvetica", 11)

    c.drawString(50, height - 210,

                 f"Customer Name: {customer_name}")

    c.drawString(50, height - 235,

                 f"Account Number: {account_number}")

    c.drawString(50, height - 260,

                 f"Contact Number: {contact_number}")


    # Feedback Section

    c.setFont("Helvetica-Bold", 14)

    c.drawString(50, height - 300,

                 "Employee Service Feedback")


    y = height - 340


    c.setFont("Helvetica-Bold", 10)

    c.drawString(60, y, "Feedback Criteria")

    c.drawString(400, y, "Rating")

    y -= 20


    ratings = []


    for q in questions:

        rating = random.randint(1, 5)

        ratings.append(rating)


        c.setFont("Helvetica", 10)

        c.drawString(60, y, q)

        c.drawString(420, y, str(rating))

        y -= 25


    # Comments

    comments = [

        "Very professional and helpful staff.",

        "Quick resolution of my issue.",

        "Satisfied with the service.",

        "Employee was courteous and knowledgeable.",

        "Overall good experience.",

        "Need improvement in response time.",

        "Excellent customer handling."

    ]


    comment = random.choice(comments)


    y -= 20

    c.setFont("Helvetica-Bold", 14)

    c.drawString(50, y, "Additional Comments")


    y -= 30

    c.setFont("Helvetica", 11)

    c.drawString(50, y, comment)


    # Recommendation

    y -= 40

    c.setFont("Helvetica-Bold", 12)

    c.drawString(

        50,

        y,

        "Would you recommend our bank to others?"

    )


    recommendation = "Yes" if sum(ratings)/len(ratings) >= 4 else "No"


    y -= 25

    c.setFont("Helvetica", 11)

    c.drawString(

        70,

        y,

        f"Recommendation: {recommendation}"

    )


    c.save()


print("100 PDF feedback forms generated successfully.")