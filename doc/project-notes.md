
- All of the lines are in boxes. The div elements of the boxes are tagged `quote`. This means that we can get all of the html `div` elements with the tag `quote` by using the xpath command `response.xpath('//div[contains(@class, "quote")]')`. Then extract the first using `response.xpath('//div[contains(@class, "quote")]').extract_first()`, etc. I think this exactly returns all the quote boxes, meaning all of the lines.
- We can do the exact same thing with css using `response.css('div.quote').extract_first()`. Again, I think it returns all of the correct things.
- Each div element will have to be dealt with in turn using something like BeautifulSoup.

- I can get the links in a list using the xpath `response.xpath('//a[contains(@href, "no")]/@href').extract()`. Takes advantage of the fact that no other element with the `a` tag has an `href` attribute containing the string `no`.
    - If I'm starting at 1-1, then should use `response.xpath('//a[contains(@href, "no")]/@href').extract()[1:]` so I don't revisit the same page twice.

### Working with raw data

- issue is that the conversation parts of the data retrieved aren't in the order of character -- character-line. If there were tags within the character line (e.g. italics), then the line is now broken over multiple entries in the list.
    - if `lines` is the json output loaded, then `lines[3]['conversation']` fives an example of a line by Pam that's broken over multiple lines
- considered writing a script to try and recognize names, count lines, and concatenate those that are really a single line, but I don't know how many breaks there could be, there could be one-off names, etc. Instead decided to just collapse everything together and then split use regex `((\w+):)`. Split a line at the largest collection letters that precede a colon `:` and that themselves are preceded by punctuation or a special character. Relies on the fact that all sentences seem to have proper punctuation after them.
    - Get names with `(\w+:)`
    - actually this doesn't work if there's a space and it seems like not all lines end with punctuation. It does look like all names start with a capital though, so now using regex `([A-Z][\w+\s]*:)`
    - still doesn't work! It's too lenient. It matches words preceding the first word if there's no punctuation. Creating a lot of weird df issues. Using `([A-Z](?:[a-z]+\s[A-Z])*[a-z]+:)` seems to work. Looks for longest string of characters with the pattern capitalized word-space (repeating), all followed by :

- there are some quote boxes on the website that are empty (e.g. 6-20). Only three though so I could manually remove them?
    - ended up using a try/except statement. If IndexError, then return an empty list. More philosophically in line with what the function does.
