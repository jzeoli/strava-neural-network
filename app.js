//require/import the HTTP module and setup express
var http = require('http');
var express = require('express')
var app = express();


//Lets define a port we want to listen to
var PORT=5000;


/*
	var strava = require('strava-v3');
		strava.athlete.listActivities({'access_token':'access_token_here'},function(err,payload) {
			console.log(payload)
		});
*/

var StravaNN = require('./neural-network')
var moment = require('moment');
var rawData = require('./run-data');
var formattedData = [];

var sdBegin = moment(new Date()).startOf('day');
var sdEnd = moment(new Date()).endOf('day');
var bottom = sdEnd.diff(sdBegin, 'second');


for(var x in rawData){

	var sd = moment(rawData[x].start_date_local);
	var sdf = moment(rawData[x].start_date_local);

	var bg = sd.startOf('day')
	var top = sdf.diff(sd, 'seconds');

	var time = top / bottom;
	var sleep = (rawData[x].sleep - 5) / 5;
	var temp = (rawData[x].temp - 48) / ((84-48));

	var obj = {input: [time, sleep, rawData[x].wy, temp], output: [rawData[x].output]}
	formattedData.push(obj)
}

//First train the model on the strava data
var model = new StravaNN();
model.learn(formattedData)

//Predict an outcome based on inputs
var prediction = model.predict([0.3, 0.8, 0, .5]);
console.log("Second Test - " +  prediction[0][0])




app.get('/', function(req, res) {
    res.sendfile('index.html', {root: __dirname })
});



//Lets start our server
app.listen(PORT, function(){
    //Callback triggered when server is successfully listening. Hurray!
    console.log("Server listening on: http://localhost:%s", PORT);
});
