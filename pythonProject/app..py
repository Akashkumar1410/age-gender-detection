from flask import Flask, request, render_template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name)

@app.route('/', methods=['GET', 'POST'])
def contact_form():
    if request.method == 'POST':
        sender_email = request.form['email']
        message = request.form['message']

        # Send the email
        send_email(sender_email, message)

    return render_template('contact.html')

def send_email(sender_email, message):
    # Setup email server and credentials
    smtp_server = 'your-smtp-server.com'
    smtp_port = 587
    smtp_username = 'your-email@example.com'
    smtp_password = 'your-email-password'

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = 'your-email@example.com'
    msg['Subject'] = 'Contact Form Submission'

    msg.attach(MIMEText(message, 'plain'))

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, 'your-email@example.com', msg.as_string())

if __name__ == '__main__':
    app.run()
