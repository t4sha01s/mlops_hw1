import os
from sqlalchemy import func
from flask import Flask, redirect, url_for, session, request, make_response, jsonify
from flask_restx import Api, Resource, Namespace, fields, marshal, marshal_with, abort
from flask_sqlalchemy import SQLAlchemy
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix

"""
Setting app configurations
"""

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(
    app, 
    title='REST API for task management',
    description="""Documentation for revolutionary AI using app to handle you day-to-day task!!!.\n\n
    Before getting to work with this application please go through authorization procedure.
    To do so add /login to your current url.
    (for example, http://localhost:5000/login)"""
 )

db = SQLAlchemy(app)

namespace = api.namespace('', 'Click the down arrow to expand the content')

model = api.model('Model', {
    'id': fields.Integer(description='Unique idetnification number, set automatically'),
    'title': fields.String(required=True, description='Name of the task'),
    'description': fields.String(description='Description of the task'),
    'done': fields.Boolean(description='Status: true if finished, false otherwise'),
})

post_model = api.model('Post model', {
    'title': fields.String(required=True, default='Task title', description='Name of the task'),
    'description': fields.String(default='Task description', description='Description of the task')
})

put_model = api.model('Put model', {
    'title': fields.String(default='New title', description='Name of the task'),
    'description': fields.String(default='New description', description='Description of the task'),
    'done': fields.Boolean(default=True, description='Status: true if finished, false otherwise')
})


"""
Initializing authorization via github
"""

oauth = OAuth(app)

github = oauth.register(
    name='github',
    client_id='Ov23liWccuX6xt5sYPwl',
    client_secret='9594d99c778b94ad8b9441937d182badb43ae795',
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

@app.route("/") 
def index(): 
    return redirect("/login")

@app.route('/login')
def registro():
    github = oauth.create_client('github')
    redirect_uri = url_for('authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    github = oauth.create_client('github')
    token = github.authorize_access_token()
    resp = github.get('user', token=token)
    profile = resp.json()
    if 'id' not in profile:
        abort(400, "GitHub authorization failed")
    github_id = profile['id']
    session['token_oauth'] = token
    session['github_id'] = profile['id']
    return redirect(url_for('index'))
    #return jsonify({'message': 'Authorization successful', 'token': token, 'github_id': github_id})    

def get_user_id():
    if 'github_id' in session:
        return session['github_id']
    return 
  
"""
Initializing database for tasks storage
"""
  
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120))
    description = db.Column(db.String(1000))
    done = db.Column(db.Boolean)
    user_id = db.Column(db.Integer, index=True)

    def __init__(self, title, description, user_id):
        self.title = title
        self.description = description
        self.done = False
        self.user_id = user_id

    def __repr__(self):
        return '<task %r>' % self.title

    def to_dict(self):
        return {'id': self.id, 'title': self.title, 'description': self.description, 'done': self.done}

with app.app_context():
    db.create_all()

"""
Handling endpoints
"""

@namespace.route('/tasks')
class TasksList(Resource):
    @namespace.doc(responses={200: 'OK'})
    @api.doc(description="Getting the list with all tasks")
    @api.marshal_list_with(model)
    def get(self):
        u_id = get_user_id()
        if not u_id:
            abort(401, 'User not authorized')
        res = []
        for el in Task.query.filter_by(user_id=u_id).all():
            res.append(el.to_dict())
        return res, 200
          
    @namespace.doc(responses={201: 'Created', 400: 'Required field \'title\' is missing'}) 
    @api.doc(description="Create new task")
    @api.expect(post_model)
    @api.marshal_list_with(model)   
    def post(self):
        u_id = get_user_id()
        if not u_id:
            abort(401, 'User not authorized')
        if not api.payload or not 'title' in api.payload:
            abort(400, 'Bad Request', extended_message='Required field \'title\' is missing')
        task_title = api.payload['title']
        task_descr =  api.payload.get('description', '')
        new_task = Task(task_title, task_descr, u_id)
        db.session.add(new_task)
        db.session.commit()
        api.payload['id'] = db.session.query(Task.id).filter(Task.id == db.session.query(func.max(Task.id))).scalar()
        api.payload['done'] = False
        return api.payload, 201

@namespace.route('/tasks/<int:task_id>')#
class TaskById(Resource):
    @namespace.doc(responses={200: 'OK', 404: 'Not found'})
    @api.doc(description="Get task by id")
    def get(self, task_id):
        u_id = get_user_id()
        if not u_id:
            abort(401, 'User not authorized')
        query = Task.query.filter_by(user_id=u_id).filter_by(id=task_id).first()
        if not isinstance(query, Task):
            abort(404, 'Not found')
        return make_response(jsonify(query.to_dict()), 200)
        
    @namespace.doc(responses={200: 'OK', 404: 'Not found'})
    @api.doc(description="Change existing task")   
    @api.expect(put_model)
    @api.marshal_list_with(model)    
    def put(self, task_id):
        u_id = get_user_id()
        if not u_id:
            abort(401, 'User not authorized')
        with app.app_context():
            query = Task.query.filter_by(user_id=u_id).filter_by(id=task_id).first()
            if not isinstance(query, Task):
                abort(404, 'Not found')
            if 'title' in request.json:
                query.title = request.json['title']
            if 'description' in request.json:
                query.description = request.json['description']
            if 'done' in request.json:
                query.done = request.json['done']
            db.session.commit()
            api.payload['id'] = task_id
            return api.payload, 200
            
    @namespace.doc(responses={204: 'No Content', 404: 'Not found'})
    @api.doc(description="Delete existing task")
    def delete(self, task_id):
        u_id = get_user_id()
        if not u_id:
            abort(401, 'User not authorized')
        with app.app_context():
            query = Task.query.filter_by(user_id=u_id).filter_by(id=task_id).first()
            if not isinstance(query, Task):
                abort(404, 'Not found')
            db.session.delete(query)
            db.session.commit()
        return make_response('', 204)     

if __name__ == '__main__':
    app.run(debug=True)





    
    
