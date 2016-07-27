fs = require 'fs'
_ = require 'lodash'
async = require 'async'

dir = fs.readdirSync("json_m")

queue = []

uuid = require 'uuid'
request = require 'request'


for filename in dir
  json = JSON.parse(fs.readFileSync('json_m/'+filename))
  items = json['items']
  for item in items
    images = [item['primary_headshot']]
    obs = item['media_objs'] || []
    for media in obs 
      if media['media_type']=='I'
        #images.push(media['media_url'])
        images.push(media['media_thumb'])
    images = _.uniq(images)
    for image in images
      queue.push(image)


downloadFile = (url, callback) ->
  req = request({method:"GET", uri:url, gzip:true})
  s = url.split('/')
  req.pipe(fs.createWriteStream("original_m/"+s[s.length-1]))
  req.on 'close', =>

  req.on 'end', =>
    callback()
  req.on 'error', =>
    console.log('error')

downloadDone = () ->
  console.log("DONE")

async.eachLimit(queue, 20, downloadFile, downloadDone)
console.log('found images', queue.length)
