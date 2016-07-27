require "open-uri"
require 'json'

1.upto(2000).each do |i|
  #uri = 'https://www.backstage.com/talent/async/search/?gender=F&max_age__gte=16&min_age__lte=45&assets=headshot_available&size=48&page='+i.to_s
  uri = 'https://www.backstage.com/talent/async/search/?gender=M&max_age__gte=16&min_age__lte=45&assets=headshot_available&size=48&page='+i.to_s
  data = open(uri, 'X-Requested-With' => 'XMLHttpRequest').read

  json = JSON.parse(data)

  File.open("json_m/"+i.to_s+".json", "w") { |f| f.write(json.to_json) }
end
