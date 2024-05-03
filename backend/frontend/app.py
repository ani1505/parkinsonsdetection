from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/')
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

def index():
    return render_template('welcome.html')

@app.route('/register',methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if not name or not email or not password:
            error = 'All fields are required'
        else:
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect('/welcome')  # Redirect to welcome page after successful registration

    return render_template('welcome.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not email or not password:
            error = 'All fields are required'
        else:
            user = User.query.filter_by(email=email).first()

            if user and user.check_password(password):
                session['email'] = user.email
                return redirect('/index')
            else:
                error = 'Invalid email or password'

    return render_template('welcome.html', error=error)



@app.route('/index')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('index.html',user=user)
    
    return redirect('/index.html')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/welcome')

if __name__ == '__main__':
    app.run(debug=True)
    
