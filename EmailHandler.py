
'''

Class designed to handle constructing and sending emails, usually in the context of notifying one
or more recipients when a process has finished. Currently, the EmailHandler has the capability
to send an email to one or more people as long as they are identified in the Address Book,
and attach one or more files to the email

@author Daniel Enrico Cahall
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
         'Dr.Bouaynaya': 'bouaynaya@rowan.edu', 
         'Oliver': 'palumb48@students.rowan.edu',
         'Dimah': 'derad6@rowan.edu',
         'Alena': 'alenagusevarus@gmail.com',
         'Dr.Hassan': 'hfshaykh@uabmc.edu',
         'Dr.Rasool': 'rasool@rowan.edu'}
    body = ""
    
    ## The constructor for the emailHandler class. This creates the message and sets the from address to the mrimathnotifier@gmail.com
    def __init__(self):
        self.msg = MIMEMultipart()
        self.msg['From']  = self.addr
    
    ## Prepares the message to be sent by setting up the subject and body
    #
    # @param subject Subject line of the email to be sent
    # @param body Body of the email to be sent
    def prepareMessage(self, subject, body):
        self.msg['Subject'] = subject
        self.body = body
    
    ## Connects to the gmail server and logs in using the mrimathnotifier gmail address
    def connectToServer(self):
        self.server = smtplib.SMTP('smtp.gmail.com', 587)
        self.server.ehlo()
        self.server.starttls()
        self.server.ehlo()
        self.server.login(self.addr, self.password)
    
    ## Sends the email to all desired recipients as long as they are within the address book
    #
    # @param recipients a list of the names of desired recipients which have their emails linked in the address book map
    def sendMessage(self, recipients):
        self.msg['To'] = ','.join(list(map(self.addressBook.get, recipients)))
        self.body = "Hello,\n\n" + "This is an automated message from the MRIMath Notifier:" + "\n\n"+ self.body
        self.body = self.body + "\n\n" + "Regards,\nMRIMath Notifier"
        self.msg.attach(MIMEText(self.body, 'plain'))
        self.server.sendmail(self.addr, list(map(self.addressBook.get, recipients)), self.msg.as_string())
    
    ## Clears the body of the email, the attached files, and the list of recipients, and disconnects from the gmail server
    def finish(self):
        self.body = ''
        if 'To' in self.msg:
            self.msg.replace_header('To', '')
        self.msg.set_payload([])
        self.server.quit()
    
    ## Attaches a file to the email
    #
    # @param file the file to attach to the email
    # @param filename the name of the file to attach (may not be necessary actually...)
    def attachFile(self, file, filename):
        attachment = open(file.name, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        self.msg.attach(part)
        
