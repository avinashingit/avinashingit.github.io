#!/usr/bin/env ruby
# Imports public, non-interview ML algorithm notes from the Obsidian vault.

require "cgi"
require "fileutils"
require "json"

source_root = ARGV.fetch(0)
repo_root = ARGV.fetch(1)
vault_root = File.expand_path("../../..", source_root)
output_root = File.join(repo_root, "notes", "ml-algorithms")
data_path = File.join(repo_root, "_data", "ml_algorithm_notes.yml")

def slugify(value)
  value
    .downcase
    .gsub("&", " and ")
    .gsub(/[^a-z0-9]+/, "-")
    .gsub(/\A-+|-+\z/, "")
end

def strip_frontmatter(text)
  text.sub(/\A(?:\uFEFF)?\s*---\s*\R.*?\R---\s*\R/m, "")
end

def remove_interview_material(text)
  blocked_heading = /interview|question bank|rapid[- ]?fire|detailed answers?/i
  blocked_line = /interview|behavioral prep|question bank|detailed answers?|\bMOC\b/i
  skipped_level = nil
  in_fence = false
  kept = []

  text.each_line do |line|
    if line.match?(/\A\s*```/)
      in_fence = !in_fence
      kept << line unless skipped_level
      next
    end

    if in_fence
      kept << line unless skipped_level
      next
    end

    heading = line.match(/\A(#+)\s+(.+?)\s*\z/)
    if heading
      level = heading[1].length
      skipped_level = nil if skipped_level && level <= skipped_level
      if skipped_level.nil? && heading[2].match?(blocked_heading)
        skipped_level = level unless level == 1
        next
      end
    end

    next if skipped_level
    next if line.match?(blocked_line)

    kept << line
  end

  kept.join
end

def clean_body(text)
  body = strip_frontmatter(text)
  body = remove_interview_material(body)
  body = body.sub(/^#\s+.*\R+/, "")
  body = body.sub(/\A\s*---\s*\R+/, "")
  body.gsub(/\n{3,}/, "\n\n").strip
end

def description_for(body)
  plain = body
    .gsub(/\$\$.*?\$\$/m, " ")
    .gsub(/```.*?```/m, " ")
    .gsub(/<[^>]+>/, " ")
    .lines
    .map(&:strip)
    .find do |line|
      !line.empty? &&
        !line.start_with?("#", "- ", "* ", "|", ">", "!", "---") &&
        !line.match?(/\A\d+\.\s/)
    end
    .to_s
    .gsub(/\[([^\]]+)\]\([^)]+\)/, '\1')
    .gsub(/[*_`]/, "")
    .gsub(/\s+/, " ")
    .strip

  plain.length > 180 ? "#{plain[0, 177].rstrip}…" : plain
end

sections = [
  {
    number: "01",
    title: "Core Concepts",
    slug: "core-concepts",
    patterns: ["Common Concepts/*.md"],
    excluded: []
  },
  {
    number: "02",
    title: "Supervised Learning",
    slug: "supervised-learning",
    patterns: ["Machine Learning/Supervised/*.md"],
    excluded: []
  },
  {
    number: "03",
    title: "Unsupervised Learning",
    slug: "unsupervised-learning",
    patterns: ["Machine Learning/UnSupervised/Clustering/*.md"],
    excluded: []
  },
  {
    number: "04",
    title: "Deep Learning",
    slug: "deep-learning",
    patterns: ["Deep Learning/*.md"],
    excluded: []
  },
  {
    number: "05",
    title: "Graph Learning",
    slug: "graph-learning",
    patterns: ["Graph Learning/*.md"],
    excluded: ["GNN Interview Questions.md", "Graph Learning MOC.md"]
  },
  {
    number: "06",
    title: "Language Models",
    slug: "language-models",
    patterns: ["Common Concepts/LLMs/*.md"],
    excluded: ["LLM Concepts MOC.md"]
  }
]

sections.each do |section|
  section[:paths] = section[:patterns]
    .flat_map { |pattern| Dir.glob(File.join(source_root, pattern)) }
    .reject { |path| section[:excluded].include?(File.basename(path)) }
    .sort_by { |path| File.basename(path, ".md").downcase }
end

note_records = []
sections.each do |section|
  section[:paths].each_with_index do |path, index|
    title = File.basename(path, ".md")
    slug = slugify(title)
    body = clean_body(File.read(path))
    next if body.empty?

    note_records << {
      path: path,
      body: body,
      filename: title,
      title: title,
      slug: slug,
      url: "/notes/ml-algorithms/#{section[:slug]}/#{slug}/",
      section: section,
      order: index + 1,
      updated: File.mtime(path)
    }
  end
end

if note_records.size != 60
  abort "Expected 60 substantive non-interview algorithm notes, found #{note_records.size}"
end

note_url_map = {}
note_records.each do |record|
  note_url_map[record[:filename]] = record[:url]
end

attachment_map = {}
Dir.glob(File.join(vault_root, "Attachments", "**", "*")).each do |path|
  attachment_map[File.basename(path)] ||= path if File.file?(path)
end

keyword_rules = {
  "Clustering" => /cluster|k-means|dbscan|hierarchical/i,
  "Deep Learning" => /deep learning|neural network|cnn|rnn|backprop/i,
  "Embeddings" => /embedding|word2vec|node2vec|representation/i,
  "Evaluation" => /evaluation|metric|ndcg|precision|recall|calibrat/i,
  "Graphs" => /graph|gnn|gcn|graphsage|message passing|pagerank/i,
  "Inference" => /inference|decoding|kv cache|serving|latency/i,
  "LLMs" => /llm|language model|rlhf|prompt|tokenization|rag\b/i,
  "Linear Models" => /linear regression|logistic regression|svm|support vector/i,
  "Optimization" => /optimi|gradient descent|optimizer|learning rate/i,
  "Probability" => /probab|bayes|likelihood|entropy/i,
  "Retrieval" => /retriev|search|nearest neighbor|rag\b/i,
  "Supervised Learning" => /supervised|classification|regression|random forest|boost/i,
  "Training" => /training|loss|regularization|fine-tun/i,
  "Transformers" => /transformer|attention|positional encod/i,
  "Trees" => /decision tree|random forest|boosted tree|xgboost/i,
  "Unsupervised Learning" => /unsupervised|cluster|dimensionality/i
}.freeze

section_keywords = {
  "01" => ["Training", "Optimization"],
  "02" => ["Supervised Learning"],
  "03" => ["Unsupervised Learning", "Clustering"],
  "04" => ["Deep Learning", "Training"],
  "05" => ["Graphs", "Embeddings"],
  "06" => ["LLMs", "Transformers"]
}.freeze

FileUtils.rm_rf(output_root)
FileUtils.mkdir_p(output_root)
FileUtils.mkdir_p(File.dirname(data_path))

note_records.each do |record|
  body = record[:body]
  source_for_keywords = "#{record[:title]}\n#{body}"
  keyword_scores = Hash.new(0)
  section_keywords.fetch(record[:section][:number]).each { |keyword| keyword_scores[keyword] += 100 }
  keyword_rules.each do |keyword, pattern|
    keyword_scores[keyword] += 8 if record[:title].match?(pattern)
    keyword_scores[keyword] += 3 if body[0, 1600].to_s.match?(pattern)
    keyword_scores[keyword] += 1 if source_for_keywords.match?(pattern)
  end
  record[:keywords] = keyword_scores
    .sort_by { |keyword, score| [-score, keyword_rules.keys.index(keyword)] }
    .first(5)
    .map(&:first)

  note_dir = File.join(output_root, record[:section][:slug], record[:slug])
  image_dir = File.join(note_dir, "images")
  FileUtils.mkdir_p(note_dir)

  body = body.gsub(/!\[\[([^\]]+)\]\]/) do
    asset_name = Regexp.last_match(1).split("|").first
    source = attachment_map[asset_name]
    unless source
      warn "Missing attachment: #{asset_name} in #{record[:filename]}"
      next "*Diagram unavailable*"
    end

    FileUtils.mkdir_p(image_dir)
    extension = File.extname(asset_name)
    destination_name = "#{slugify(File.basename(asset_name, extension))}#{extension.downcase}"
    FileUtils.cp(source, File.join(image_dir, destination_name))
    alt = File.basename(asset_name, extension).sub(/\s+mermaid\s+\d+\z/i, "")
    "![#{alt}](images/#{destination_name})"
  end

  body = body.gsub(/\[\[#([^\]|]+)(?:\|([^\]]+))?\]\]/) do
    heading = Regexp.last_match(1)
    label = Regexp.last_match(2) || heading
    "[#{label}](##{slugify(heading)})"
  end

  body = body.gsub(/\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]/) do
    target = Regexp.last_match(1)
    heading = Regexp.last_match(2)
    label = Regexp.last_match(3) || File.basename(target)
    url = note_url_map[target] || note_url_map[File.basename(target)]
    if url
      url += "##{slugify(heading)}" if heading
      "[#{label}](#{url})"
    else
      label
    end
  end

  body = body.gsub(/^> \[!(\w+)\][+-]?(?:\s+(.+))?$/) do
    kind = Regexp.last_match(1).capitalize
    text = Regexp.last_match(2)
    text ? "> **#{kind}:** #{text}" : "> **#{kind}**"
  end

  body = body.gsub(/^```mermaid\s*\n(.*?)^```\s*$/m) do
    diagram = CGI.escapeHTML(Regexp.last_match(1).strip)
    "<pre class=\"mermaid\">\n#{diagram}\n</pre>"
  end

  if body.match?(/interview/i)
    abort "Interview material remained after sanitizing #{record[:filename]}"
  end

  record[:description] = description_for(body)
  has_mermaid = body.include?('class="mermaid"')
  frontmatter = [
    "---",
    "layout: note",
    "title: #{JSON.generate(record[:title])}",
    "description: #{JSON.generate(record[:description])}",
    "note: true",
    "note_collection: \"ML algorithms\"",
    "note_section: #{JSON.generate(record[:section][:title])}",
    "section_order: #{record[:section][:number].to_i}",
    "note_order: #{record[:order]}",
    "updated: #{record[:updated].strftime("%Y-%m-%d %H:%M:%S %z")}",
    "keywords:",
    *record[:keywords].map { |keyword| "  - #{keyword}" },
    "math: true",
    "mermaid: #{has_mermaid}",
    "---",
    ""
  ].join("\n")

  File.write(File.join(note_dir, "index.md"), frontmatter + body.strip + "\n")
end

data_lines = [
  "title: \"ML algorithms\"",
  "keywords:"
]
note_records.flat_map { |record| record[:keywords] }.uniq.sort.each do |keyword|
  data_lines << "  - #{JSON.generate(keyword)}"
end
data_lines << "notes:"
note_records.sort_by { |record| [-record[:updated].to_i, record[:section][:number], record[:order]] }.each do |record|
  data_lines.concat([
    "  - title: #{JSON.generate(record[:title])}",
    "    url: #{JSON.generate(record[:url])}",
    "    section: #{JSON.generate(record[:section][:title])}",
    "    section_number: #{JSON.generate(record[:section][:number])}",
    "    description: #{JSON.generate(record[:description])}",
    "    updated: #{JSON.generate(record[:updated].strftime("%Y-%m-%d"))}",
    "    keywords:",
    *record[:keywords].map { |keyword| "      - #{JSON.generate(keyword)}" }
  ])
end

data_lines << "sections:"
sections.each do |section|
  data_lines.concat([
    "  - number: #{JSON.generate(section[:number])}",
    "    title: #{JSON.generate(section[:title])}",
    "    slug: #{JSON.generate(section[:slug])}",
    "    notes:"
  ])
  note_records.select { |record| record[:section] == section }.each do |record|
    data_lines.concat([
      "      - title: #{JSON.generate(record[:title])}",
      "        url: #{JSON.generate(record[:url])}"
    ])
  end
end

File.write(data_path, data_lines.join("\n") + "\n")

puts "Imported #{note_records.size} ML algorithm notes across #{sections.size} sections."
