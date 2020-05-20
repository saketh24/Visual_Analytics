var margin = {top: 30, right: 20, bottom: 30, left: 50},
    width = 600 - margin.left - margin.right,
    height = 270 - margin.top - margin.bottom;
function draw_scree(link){
	var title;
	if(link === "/scree_random"){
		title = "Scree Plot Random Samples";
	}
	else if (link === "/scree_stratified"){
		title = "Scree Plot Stratified Samples";
	}
	else if (link === "/scree_original"){
		title = "Scree Plot Original Data";
	}
	var data;
    $.ajax({
	  type: 'GET',
	  url:link,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(res) {
	  	make_scree_plot(res,title);
	  },
	  error: function(res) {
		$("#chart_container").html(res);
	  }
	});
}

function draw_scatter(link){
	var title;
	if(link === "/pca_random_data"){
		title = "PCA Scatter Plot Random Samples";
	}
	else if (link === "/pca_stratified_data"){
		title = "PCA Scatter Plot Stratified Samples";
	}
	else if (link === "/pca_original_data"){
		title = "PCA Scatter Plot Original Data";
	}
	else if (link === "/mds_euclid_random_data"){
		title = "MDS Euclidian Scatter Plot Random Data";
	}
	else if (link === "/mds_euclid_strat_data"){
		title = "MDS Euclidian Scatter Plot Stratified Data";
	}
	else if (link === "/mds_euclid_orig_data"){
		title = "MDS Euclidian Scatter Plot Original Data";
	}
	else if (link === "/mds_corr_random_data"){
		title = "MDS Correlation Scatter Plot Random Data";
	}
	else if (link === "/mds_corr_strat_data"){
		title = "MDS Correlation Scatter Plot Stratified Data";
	}
	else if (link === "/mds_corr_orig_data"){
		title = "MDS Correlation Scatter Plot Original Data";
	}
	$.ajax({
	  type: 'GET',
	  url:link,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(res) {
	  	make_scatter_plot(res, title);
	  },
	  error: function(res) {
		$("#chart_container").html(res);
	  }
	});
}

function draw_scatter_matrix(link) {
	var title;
	if(link === "/matrix_plot_random"){
		title = "Matrix Scatter Plot Random Samples";
	}
	else if (link === "/matrix_plot_strat"){
		title = "Matrix Scatter Plot Stratified Samples";
	}
	else if (link === "/matrix_plot_orig"){
		title = "Matrix Scatter Plot Original Data";
	}
	$.ajax({
	  type: 'GET',
	  url:link,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(res) {
	  	make_matrix_scatter_plot(res, title);
	  },
	  error: function(res) {
		$("#chart_container").html(res);
	  }
	});

}

function make_scree_plot(result, title){
	d3.select('#chart').remove();
	var data = JSON.parse(result);
	var eigen_vals = data['chart_data']['eigen_values'];
	var explained_variance = data['chart_data']['variance'];
	var margin = {top: 20, right: 20, bottom: 30, left: 60};
    var width = 1300 - margin.left - margin.right;
    var height = 450 - margin.top - margin.bottom;

    var chart_width = 800;
    var chart_height = 550;

    var x = d3.scale.linear().domain([1, explained_variance.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scale.linear().domain([0, d3.max(explained_variance)]).range([height, 0]);

    var xAxis = d3.svg.axis().scale(x).orient("bottom").ticks(16);
    var yAxis = d3.svg.axis().scale(y).orient("left");
    var circle_X,circle_Y;
	var line = d3.svg.line()
		.x(function (d,i) {
		if (i === 6) {
			circle_X = x(i);
			circle_Y = y(d)
            }
			return x(i);})
		.y(function(d){console.log(y(d));return y(d);});
    var svg = d3.select("body").append("svg")
		.attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", chart_height)
        .append("g")
        .attr("transform", "translate(250,10)");
    svg.append("g")
		.attr("class", "x_axis")
        .attr("transform", "translate(100,400)")
        .call(xAxis);
    svg.append("g")
		.attr("class", "y_axis")
        .attr("transform", "translate(100,0)")
        .call(yAxis);

    svg.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ 50 +","+(height/2)+")rotate(-90)")
        .text("Eigen Values");

    svg.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ ((chart_width/2) + 50) +","+ ((chart_width/2) + 50) +")")
        .text("PC");
    svg.append("path")
		.attr("fill", "none")
		.attr("stroke", "steelblue")
		.attr("stroke-width", 1.5)
        .attr("d", line(explained_variance))
        .attr("transform", "translate(150,0)");
	svg.append("circle")
		.attr("cx", circle_X)
        .attr("cy", circle_Y)
		.attr("r", 6)
		.attr("transform", "translate(90,30)")
		.style("fill", "red")
		.style("stroke", "red");
    svg.selectAll("bar")
		.data(eigen_vals)
		.enter().append("rect")
		.attr("class","bar")
		.attr("x", function (d,i) {return x(i);})
		.attr("y",function (d) {return y(d);})
		.attr("height", function (d) {return height - y(d);})
		.attr("width", 20)
		.attr("fill","steelblue")
		.attr("transform","translate(145,0)");
    svg.append("text")
        .attr("class", "chart_name")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(400,0)")
        .text(title);
}

function make_scatter_plot(result,title){
	d3.select('#chart').remove();
	var data = JSON.parse(result.chart_data);
	data.forEach(function(d) {
      d.x = +d['0'];
      d.y = +d['1'];
      });

	var margin = {top: 20, right: 20, bottom: 30, left: 60};
    var width = 1300 - margin.left - margin.right;
    var height = 450 - margin.top - margin.bottom;

    var chart_width = 800;
    var chart_height = 550;
    var xValue = function(d) { return d.x;};
    var xScale = d3.scale.linear().range([0, chart_width]);
    var xMap = function(d) { return xScale(xValue(d)); };
    var xAxis = d3.svg.axis().scale(xScale).orient("bottom");

    var yValue = function(d) { return d.y;};
    var yScale = d3.scale.linear().range([height, 0]);
    var yMap = function(d) { return yScale(yValue(d));};
    var yAxis = d3.svg.axis().scale(yScale).orient("left");
    xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
    yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);
    var svg = d3.select("body").append("svg")
		.attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", chart_height)
        .append("g")
        .attr("transform", "translate(250,10)");
	svg.append("g")
		.attr("transform", "translate(0," + height + ")")
		.attr("class", "x_axis")
		.call(xAxis);

	svg.append("g")
		.attr("class", "y_axis")
		.call(yAxis);

    svg.append("text")
            .attr("class", "axis_label")
            .attr("text-anchor", "middle")
            .attr("transform", "translate("+ (-70) +","+(height/2)+")rotate(-90)")
            .text("PCA Component 2");

    svg.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate("+ (chart_width/2) +","+(height + margin.top + margin.bottom)+")")
        .text("PCA Component 1");

    svg.selectAll('.dot')
		.data(data)
		.enter().append("circle")
		.attr("r",3.5)
		.attr("cx", xMap)
		.attr("cy", yMap)
		.style("fill","steelblue");
    svg.append("text")
        .attr("class", "chart_name")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(400,0)")
        .text(title);
}

function make_matrix_scatter_plot(result,title){
	d3.select('#chart').remove();
	var data = JSON.parse(result.chart_data);
	var cols = Object.keys(data);
	console.log(cols);
	var width = 960;
    var size = 170;
    var padding = 10;

    var chart_width = 1300;
    var chart_height = 560;

    var x = d3.scale.linear().range([padding/2, size - padding/2]);
    var y = d3.scale.linear().range([size - padding/2, padding/2]);

    var xAxis = d3.svg.axis().orient("bottom").scale(x).ticks(6).tickSize(size * 3);
    var yAxis = d3.svg.axis().orient("left").scale(y).ticks(6).tickSize(-size * 3);
    var domains = {};

    cols.forEach(function (t) {
		domains[t] = d3.extent(d3.values(data[t]));
	});

    var svg = d3.select("body").append("svg")
      .attr("id", "chart")
      .attr("width", chart_width)
      .attr("height", chart_height)
      .append("g")
      .attr("transform", "translate(400,20)");

    svg.selectAll(".x.axis")
      .data(cols)
      .enter().append("g")
      .attr("class", "x_axis")
      .attr("transform", function(d, i) { return "translate(" + (3 - i - 1) * size + ",0)"; })
      .each(function(d) { x.domain(domains[d]); d3.select(this).call(xAxis); });

    svg.selectAll(".y.axis")
      .data(cols)
      .enter().append("g")
      .attr("class", "y_axis_scatter")
      .attr("transform", function(d, i) { return "translate(0," + i * size + ")"; })
      .each(function(d) { y.domain(domains[d]); d3.select(this).call(yAxis); });

    var cell = svg.selectAll(".cell")
      .data(cross(cols, cols))
      .enter().append("g")
      .attr("class", "cell")
      .attr("transform", function(d) { return "translate(" + (3 - d.i - 1) * size + "," + d.j * size + ")"; })
      .each(plot);

    svg.append("text")
	  .attr("class", "scree_name")
      .attr("text-anchor", "middle")
      .attr("transform", "translate(650,0)")
      .text(title);

	cell.filter(function(d) { return d.i === d.j; }).append("text")
      .attr("x", padding)
      .attr("y", padding)
      .attr("dy", ".71em")
      .text(function(d) { return d.x; });

	function plot(p){
		var cell = d3.select(this);
		x.domain(domains[String(p.x)]);
		y.domain(domains[String(p.y)]);
		cell.append("rect")
			.attr("class", "frame")
			.attr("x", padding/2)
			.attr("y", padding/2)
			.attr("width", size - padding)
			.attr("height", size - padding)
			.attr('fill','white');
		c1 = data[String(p.x)];
		c2 = data[String(p.y)];
		x2 = d3.values(c2);
		var final_array = [];
		d3.values(c1).forEach(function(d, i) {
              temp_map = {};
              temp_map["x"] = d;
              temp_map["y"] = x2[i];
              final_array.push(temp_map);
          });
		cell.selectAll("circle")
	          .data(final_array)
              .enter().append("circle")
              .attr("cx", function(d) { return x(d.x); })
              .attr("cy", function(d) { return y(d.y); })
              .attr("r", 4)
              .style("fill", "steelblue");
	}
}
function cross(f1, f2) {
  var mat = [], len1 = f1.length, len2 = f2.length, i, j;
  for (i = 0; i < len1; i++)
    for (j = 0; j < len2; j++)
      mat.push(
        {x: f1[i], i: i,
          y: f2[j], j: j});
  return mat;
}