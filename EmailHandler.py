'''
Created on Dec 27, 2017

@author: daniel
'''

import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class EmailHandler:
    addr = "mrimathnotifier@gmail.com"
    password = "mrimathpw"
    addressBook = {'Danny': 'danielenricocahall@gmail.com', 
         "Daniel": 'cahalld0@students.rowan.edu',
         'Dr.Bouaynaya': 'bouaynaya@rowan.edu', 
         'Oliver': 'palumb48@students.rowan.edu',
         'Dimah': 'derad6@rowan.edu',
         'Alena': 'alenagusevarus@gmail.com',
         'Dr. Hassan': 'hfshaykh@uabmc.edu'}
    body = ""
    
    def __init__(self):
        self.msg = MIMEMultipart()
        self.msg['From']  = self.addr
    
    def prepareMessage(self, subject, body):
        self.msg['Subject'] = subject
        self.body = body
        
    def connectToServer(self):
        self.server = smtplib.SMTP('smtp.gmail.com', 587)
        self.server.ehlo()
        self.server.starttls()
        self.server.ehlo()
        self.server.login(self.addr, self.password)
        
    def sendMessage(self, recipients):
        recipient_string = ''
        for recipient in recipients:
            recipient_string += self.addressBook.get(recipient) + ", "
        print(recipient_string)
        self.msg['To'] = recipient_string
        self.body = "Hello,\n\n" + self.body
        self.body = self.body + "\n\n" + "Regards,\nMRIMath Notifier"
        self.msg.attach(MIMEText(self.body, 'plain'))
        self.server.sendmail(self.addr, map(self.addressBook, recipients), self.msg.as_string())
    
    def finish(self):
        self.body = ''
        if 'To' in self.msg:
            self.msg.replace_header('To', '')
        self.msg.set_payload([])
        self.server.quit()
        
    def attachFile(self, file, filename):
        attachment = open(file.name, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        self.msg.attach(part)
        
